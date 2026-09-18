import SwiftUI
import ARKit
import CaptureRules

/// Outlines and labels of what the phone recognises, drawn over the camera view.
///
/// Every outline lives in the room (its vertices are world points), and is
/// drawn through the camera as it is right now, thirty times a second: when
/// the phone turns, the outline stays on the thing and leaves the screen with
/// it, instead of hanging where the thing was when the frame was analysed.
/// Outlines only: a glowing line around each thing, nothing filled in.
struct OutlineOverlay: View {
    @ObservedObject var state: CaptureState
    let session: ARSession
    let spec: DetectionSpec
    /// Outlines from a frame older than this are not drawn (the analysis stalled, or the view is new).
    static let staleSeconds: Double = 1.5

    var body: some View {
        GeometryReader { geometry in
            TimelineView(.animation(minimumInterval: 1.0 / 30)) { _ in
                Canvas { context, size in
                    guard state.showOutlines, let frame = session.currentFrame else { return }
                    let camera = frame.camera
                    let toCamera = camera.transform.inverse
                    /// A world point on this screen, nil when it is behind the camera.
                    func onScreen(_ p: SIMD3<Float>) -> CGPoint? {
                        guard (toCamera * SIMD4(p, 1)).z < -0.05 else { return nil }
                        return camera.projectPoint(p, orientation: .portrait, viewportSize: size)
                    }
                    let bounds = CGRect(origin: .zero, size: size).insetBy(dx: -40, dy: -20)

                    // Placed furniture: a label above each box that is in view and not
                    // outlined right now (its outline carries the label then).
                    let fresh = frame.timestamp - state.regionsTime < Self.staleSeconds
                    let regions = fresh ? state.regions : []
                    let outlined = Set(regions.compactMap(\.objectId))
                    for object in state.map.objects where !outlined.contains(object.id) {
                        let top = object.center + SIMD3(0, object.size.y / 2 + 0.05, 0)
                        guard (toCamera * SIMD4(top, 1)).z < -0.3, let p = onScreen(top), bounds.contains(p) else { continue }
                        label(object.label, at: p, color: color(object.group), small: true, in: &context, size: size)
                    }

                    for region in regions {
                        // All of the outline must be in front of the camera; a thing half behind it is not drawn.
                        let points = region.worldOutline.compactMap(onScreen)
                        guard points.count >= 3, points.count == region.worldOutline.count else { continue }
                        var path = Path()
                        path.move(to: points[0])
                        for p in points.dropFirst() { path.addLine(to: p) }
                        path.closeSubpath()
                        guard path.boundingRect.intersects(CGRect(origin: .zero, size: size)) else { continue }
                        let tint = color(region.group)
                        let surface = region.group == "structure"
                        // A neon line: a wide faint halo, a tighter glow, the bright line itself.
                        let width: CGFloat = surface ? 1.5 : 2.5
                        if !surface {
                            context.stroke(path, with: .color(tint.opacity(0.16)), style: StrokeStyle(lineWidth: width + 7, lineJoin: .round))
                            context.stroke(path, with: .color(tint.opacity(0.32)), style: StrokeStyle(lineWidth: width + 3, lineJoin: .round))
                        }
                        context.stroke(path, with: .color(tint.opacity(surface ? 0.7 : 1)), style: StrokeStyle(lineWidth: width, lineJoin: .round))
                        // A label at the centre of anything large enough to read.
                        if region.share > 0.015, let c = region.worldCentroid, let centre = onScreen(c), bounds.contains(centre) {
                            label(region.label, at: centre, color: tint, small: false, in: &context, size: size)
                        }
                    }
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
        .allowsHitTesting(false)
    }

    private func color(_ group: String) -> Color {
        Color(hex: spec.groups[group]?.color ?? "#FFFFFF")
    }

    private func label(_ string: String, at point: CGPoint, color: Color, small: Bool,
                       in context: inout GraphicsContext, size: CGSize) {
        let text = context.resolve(Text(string).font(small ? .caption2.weight(.bold) : .caption.weight(.bold)).foregroundColor(.white))
        let textSize = text.measure(in: size)
        let pad: CGFloat = small ? 5 : 6
        let box = CGRect(x: point.x - textSize.width / 2 - pad, y: point.y - textSize.height / 2 - pad / 2,
                         width: textSize.width + 2 * pad, height: textSize.height + pad)
        context.fill(Path(roundedRect: box, cornerRadius: pad), with: .color(color.opacity(0.88)))
        context.draw(text, at: point)
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
