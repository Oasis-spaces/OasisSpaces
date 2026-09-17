import SwiftUI

/// Top-down view of the recording: scanned points, the path walked, where the
/// phone is and which way it looks, and a ring of directions already filmed.
struct CoverageMap: View {
    @ObservedObject var state: CaptureState

    var body: some View {
        Canvas { context, size in
            let centre = CGPoint(x: size.width / 2, y: size.height / 2)
            let ringRadius = min(size.width, size.height) / 2 - 6
            let sectorCount = 12

            // Directions filmed, as a ring of sectors.
            for sector in 0..<sectorCount {
                let a0 = Angle.degrees(Double(sector) * 30 - 90)
                let a1 = Angle.degrees(Double(sector + 1) * 30 - 90 - 3)
                var arc = Path()
                arc.addArc(center: centre, radius: ringRadius, startAngle: a0, endAngle: a1, clockwise: false)
                context.stroke(arc, with: .color(state.sectors.contains(sector) ? .green : .white.opacity(0.18)),
                               lineWidth: 5)
            }

            // Fit points and path around the phone.
            let scale = mapScale(radius: ringRadius - 8)
            func toMap(_ p: SIMD2<Float>) -> CGPoint {
                CGPoint(x: centre.x + CGFloat(p.x - state.position.x) * scale,
                        y: centre.y + CGFloat(p.y - state.position.y) * scale)
            }
            for p in state.mapPoints {
                let q = toMap(p)
                context.fill(Path(ellipseIn: CGRect(x: q.x - 0.8, y: q.y - 0.8, width: 1.6, height: 1.6)),
                             with: .color(.white.opacity(0.55)))
            }
            if state.path.count > 1 {
                var walk = Path()
                walk.move(to: toMap(state.path[0]))
                for p in state.path.dropFirst() { walk.addLine(to: toMap(p)) }
                context.stroke(walk, with: .color(.cyan), lineWidth: 2)
            }

            // The phone and its view direction.
            let heading = Angle.degrees(state.heading - 90)
            var cone = Path()
            cone.move(to: centre)
            cone.addArc(center: centre, radius: 22, startAngle: heading - .degrees(28),
                        endAngle: heading + .degrees(28), clockwise: false)
            cone.closeSubpath()
            context.fill(cone, with: .color(.yellow.opacity(0.45)))
            context.fill(Path(ellipseIn: CGRect(x: centre.x - 4, y: centre.y - 4, width: 8, height: 8)),
                         with: .color(.yellow))
        }
        .background(.black.opacity(0.45), in: Circle())
        .overlay(Circle().stroke(.white.opacity(0.15)))
    }

    /// Pixels per metre, so everything recorded fits inside the ring (at least 4 m across).
    private func mapScale(radius: CGFloat) -> CGFloat {
        var extent: Float = 2
        for p in state.path { extent = max(extent, abs(p.x - state.position.x), abs(p.y - state.position.y)) }
        return radius / CGFloat(extent)
    }
}
