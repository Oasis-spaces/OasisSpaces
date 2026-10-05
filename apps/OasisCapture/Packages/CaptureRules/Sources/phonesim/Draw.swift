// Pictures of what the simulated phone saw: each frame with its outlines and
// labels, and a top-down map of the placed boxes against the room's measured
// furniture (--dump <folder>).
import CoreGraphics
import CoreImage
import CoreText
import Foundation
import ImageIO
import simd
import CaptureRules

enum Draw {
    static let rgb = CGColorSpaceCreateDeviceRGB()

    /// A steady colour per class.
    static func color(_ index: Int, alpha: CGFloat = 1) -> CGColor {
        let hue = Double((index * 47) % 360) / 360
        let (r, g, b) = hsv(hue, 0.85, 1)
        return CGColor(colorSpace: rgb, components: [r, g, b, alpha])!
    }

    static func hsv(_ h: Double, _ s: Double, _ v: Double) -> (CGFloat, CGFloat, CGFloat) {
        let i = Int(h * 6) % 6, f = h * 6 - Double(Int(h * 6))
        let p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s)
        switch i {
        case 0: return (v, t, p); case 1: return (q, v, p); case 2: return (p, v, t)
        case 3: return (p, q, v); case 4: return (t, p, v); default: return (v, p, q)
        }
    }

    static func gray(_ v: CGFloat, _ alpha: CGFloat = 1) -> CGColor { CGColor(colorSpace: rgb, components: [v, v, v, alpha])! }

    static func text(_ string: String, at point: CGPoint, in ctx: CGContext, color: CGColor, size: CGFloat = 13) {
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, size, nil)
        let attributed = NSAttributedString(string: string, attributes: [
            kCTFontAttributeName as NSAttributedString.Key: font,
            kCTForegroundColorAttributeName as NSAttributedString.Key: color,
        ])
        let line = CTLineCreateWithAttributedString(attributed)
        var ascent: CGFloat = 0, descent: CGFloat = 0
        let width = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, nil))
        let box = CGRect(x: point.x - 2, y: point.y - descent - 1, width: width + 4, height: ascent + descent + 2)
        ctx.setFillColor(gray(0, 0.65))
        ctx.fill(box)
        ctx.textPosition = point
        CTLineDraw(line, ctx)
    }

    static func write(_ ctx: CGContext, to url: URL, jpeg: Bool) {
        guard let image = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, (jpeg ? "public.jpeg" : "public.png") as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(dest, image, jpeg ? [kCGImageDestinationLossyCompressionQuality: 0.82] as CFDictionary : nil)
        CGImageDestinationFinalize(dest)
    }

    /// The frame with every instance's outline (solid), the detector's box
    /// (dashed) and "label confidence [quality] track" in the box's corner.
    static func frame(_ image: CIImage, context: CIContext, instances: [Instance], matches: [ObservationMatch?],
                      notes: [String] = [], spec: ObjectSpec, to url: URL, maxWidth: Int = 720) {
        guard let cg = context.createCGImage(image, from: image.extent) else { return }
        let scale = min(1, CGFloat(maxWidth) / CGFloat(cg.width))
        let w = Int(CGFloat(cg.width) * scale), h = Int(CGFloat(cg.height) * scale)
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: rgb,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        // Normalised image coordinates (y down) to the context (origin bottom-left).
        func at(_ x: Double, _ y: Double) -> CGPoint { CGPoint(x: x * Double(w), y: (1 - y) * Double(h)) }
        for (i, inst) in instances.enumerated() {
            let color = Self.color(inst.classIndex)
            ctx.setStrokeColor(color)
            ctx.setLineWidth(2.5)
            let polygon = MaskOutline.polygon(of: inst.mask, width: inst.maskWidth, height: inst.maskHeight, epsilon: 1.0)
            if polygon.count >= 3 {
                ctx.beginPath()
                ctx.move(to: at(polygon[0].x, polygon[0].y))
                for p in polygon.dropFirst() { ctx.addLine(to: at(p.x, p.y)) }
                ctx.closePath()
                ctx.strokePath()
            }
            ctx.setLineWidth(1)
            ctx.setLineDash(phase: 0, lengths: [4, 4])
            ctx.stroke(CGRect(x: Double(inst.minX) * Double(w), y: (1 - Double(inst.maxY)) * Double(h),
                              width: Double(inst.maxX - inst.minX) * Double(w), height: Double(inst.maxY - inst.minY) * Double(h)))
            ctx.setLineDash(phase: 0, lengths: [])
            let label = spec.info(inst.classIndex)?.label ?? "?"
            var caption = String(format: "%@ %.0f%%", label, inst.confidence * 100)
            if let q = inst.quality { caption += String(format: " q%.2f", q) }
            if i < notes.count { caption += " " + notes[i] }
            if i < matches.count, let m = matches[i] {
                caption += " " + m.objectID + (m.placed ? "*" : "")
                if m.label != label { caption += " as \(m.label)" }
            }
            text(caption, at: CGPoint(x: Double(inst.minX) * Double(w) + 3, y: (1 - Double(inst.minY)) * Double(h) - 15), in: ctx, color: color)
        }
        write(ctx, to: url, jpeg: true)
    }

    /// The room from above: walls grey, measured furniture green, placed boxes
    /// in their class colour with "label height".
    static func map(_ map: RoomMap, truth: [Truth.Object], walls: [Truth.Wall], to url: URL, pixelsPerMetre: CGFloat = 160) {
        var lo = SIMD2<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for wall in walls {
            let c = SIMD2(wall.center[0], wall.center[2]), d = SIMD2(wall.along[0], wall.along[2]) * wall.half
            lo = simd_min(lo, simd_min(c - d, c + d)); hi = simd_max(hi, simd_max(c - d, c + d))
        }
        for o in truth { lo = simd_min(lo, SIMD2(o.min[0], o.min[2])); hi = simd_max(hi, SIMD2(o.max[0], o.max[2])) }
        for o in map.objects { lo = simd_min(lo, o.footprintMin); hi = simd_max(hi, o.footprintMax) }
        guard lo.x < hi.x else { return }
        lo -= 0.4; hi += 0.4
        let w = Int(CGFloat(hi.x - lo.x) * pixelsPerMetre), h = Int(CGFloat(hi.y - lo.y) * pixelsPerMetre)
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: rgb,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        ctx.setFillColor(gray(1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        func at(_ p: SIMD2<Float>) -> CGPoint { CGPoint(x: CGFloat(p.x - lo.x) * pixelsPerMetre, y: CGFloat(p.y - lo.y) * pixelsPerMetre) }
        // Metre grid.
        ctx.setStrokeColor(gray(0.92)); ctx.setLineWidth(1)
        var gx = lo.x.rounded(.up)
        while gx < hi.x { ctx.move(to: at(SIMD2(gx, lo.y))); ctx.addLine(to: at(SIMD2(gx, hi.y))); gx += 1 }
        var gz = lo.y.rounded(.up)
        while gz < hi.y { ctx.move(to: at(SIMD2(lo.x, gz))); ctx.addLine(to: at(SIMD2(hi.x, gz))); gz += 1 }
        ctx.strokePath()
        ctx.setStrokeColor(gray(0.35)); ctx.setLineWidth(5)
        for wall in walls {
            let c = SIMD2(wall.center[0], wall.center[2]), d = SIMD2(wall.along[0], wall.along[2]) * wall.half
            ctx.move(to: at(c - d)); ctx.addLine(to: at(c + d))
        }
        ctx.strokePath()
        let green = CGColor(colorSpace: rgb, components: [0.1, 0.6, 0.2, 1])!
        ctx.setStrokeColor(green); ctx.setLineWidth(2)
        for o in truth {
            let a = at(SIMD2(o.min[0], o.min[2])), b = at(SIMD2(o.max[0], o.max[2]))
            ctx.stroke(CGRect(x: a.x, y: a.y, width: b.x - a.x, height: b.y - a.y))
            text(String(format: "%@ %@ h%.1f", o.id, o.label, o.max[1] - o.min[1]), at: CGPoint(x: a.x + 3, y: b.y - 15), in: ctx, color: green, size: 11)
        }
        for o in map.objects {
            let color = Self.color(o.classId)
            ctx.setStrokeColor(color); ctx.setFillColor(Self.color(o.classId, alpha: 0.15)); ctx.setLineWidth(2)
            let corners = o.footprint.map(at)
            ctx.beginPath(); ctx.move(to: corners[0])
            for c in corners.dropFirst() { ctx.addLine(to: c) }
            ctx.closePath(); ctx.drawPath(using: .fillStroke)
            let centre = at(SIMD2(o.center.x, o.center.z))
            text(String(format: "%@ h%.1f %@", o.label, o.size.y, o.id), at: CGPoint(x: centre.x - 20, y: centre.y - 5), in: ctx, color: color, size: 11)
        }
        write(ctx, to: url, jpeg: false)
    }
}
