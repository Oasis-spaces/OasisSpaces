import SwiftUI
import ARKit
import CaptureRules

/// Outlines and labels of what the segmentation found, drawn over the camera view.
struct OutlineOverlay: View {
    @ObservedObject var state: CaptureState
    let session: ARSession
    let spec: DetectionSpec

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                guard state.showOutlines, !state.regions.isEmpty,
                      let frame = session.currentFrame else { return }
                // Image coordinates (sensor space, normalised) to this view's.
                let transform = frame.displayTransform(for: .portrait, viewportSize: size)
                func toView(_ p: SIMD2<Double>) -> CGPoint {
                    let n = CGPoint(x: p.x, y: p.y).applying(transform)
                    return CGPoint(x: n.x * size.width, y: n.y * size.height)
                }
                // Placed furniture: a label above each box that is in view.
                for object in state.map.objects {
                    let top = object.center + SIMD3(0, object.size.y / 2 + 0.05, 0)
                    let p = frame.camera.projectPoint(top, orientation: .portrait, viewportSize: size)
                    let local = frame.camera.transform.inverse * SIMD4(top, 1)
                    guard local.z < -0.3, p.x > -40, p.y > -20, p.x < size.width + 40, p.y < size.height + 20 else { continue }
                    let color = Color(hex: spec.groups[object.group]?.color ?? "#FFFFFF")
                    let text = context.resolve(Text(object.label).font(.caption2.weight(.bold)).foregroundColor(.white))
                    let textSize = text.measure(in: size)
                    let box = CGRect(x: p.x - textSize.width / 2 - 5, y: p.y - textSize.height / 2 - 2,
                                     width: textSize.width + 10, height: textSize.height + 4)
                    context.fill(Path(roundedRect: box, cornerRadius: 5), with: .color(color.opacity(0.9)))
                    context.draw(text, at: p)
                }
                for region in state.regions {
                    guard region.outline.count >= 3 else { continue }
                    let color = Color(hex: spec.groups[region.group]?.color ?? "#FFFFFF")
                    var path = Path()
                    path.move(to: toView(region.outline[0]))
                    for p in region.outline.dropFirst() { path.addLine(to: toView(p)) }
                    path.closeSubpath()
                    let big = region.group == "structure"
                    context.fill(path, with: .color(color.opacity(big ? 0.06 : 0.14)))
                    context.stroke(path, with: .color(color.opacity(0.95)),
                                   style: StrokeStyle(lineWidth: big ? 1.5 : 2.5, lineJoin: .round))
                    // Label at the centre of anything large enough to read.
                    if region.share > 0.015 {
                        let centre = toView(region.centroid)
                        let text = context.resolve(Text(region.label).font(.caption.weight(.bold)).foregroundColor(.white))
                        let textSize = text.measure(in: size)
                        let box = CGRect(x: centre.x - textSize.width / 2 - 6, y: centre.y - textSize.height / 2 - 3,
                                         width: textSize.width + 12, height: textSize.height + 6)
                        context.fill(Path(roundedRect: box, cornerRadius: 6), with: .color(color.opacity(0.85)))
                        context.draw(text, at: centre)
                    }
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
        .allowsHitTesting(false)
    }
}

extension Color {
    /// "#RRGGBB".
    init(hex: String) {
        var value: UInt64 = 0
        Scanner(string: hex.trimmingCharacters(in: CharacterSet(charactersIn: "#"))).scanHexInt64(&value)
        self.init(red: Double((value >> 16) & 0xFF) / 255, green: Double((value >> 8) & 0xFF) / 255,
                  blue: Double(value & 0xFF) / 255)
    }
}
