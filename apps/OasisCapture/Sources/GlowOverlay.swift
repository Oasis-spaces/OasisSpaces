import SwiftUI
import ARKit

/// The glowing class-coloured edges, laid exactly over the camera image.
struct GlowOverlay: View {
    @ObservedObject var state: CaptureState
    let session: ARSession

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                guard state.showOutlines, let glow = state.glow, let frame = session.currentFrame else { return }
                // The glow is in the sensor image's own orientation; ARKit says how
                // that image sits on this screen (rotated, cropped to fill).
                let transform = frame.displayTransform(for: .portrait, viewportSize: size)
                context.transform = transform.concatenating(CGAffineTransform(scaleX: size.width, y: size.height))
                context.blendMode = .plusLighter
                context.opacity = 0.9
                context.draw(Image(decorative: glow, scale: 1), in: CGRect(x: 0, y: 0, width: 1, height: 1))
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
        .allowsHitTesting(false)
    }
}
