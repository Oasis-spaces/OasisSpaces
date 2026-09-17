import SwiftUI
import MetalKit
import MetalSplatter
import SplatIO
import simd

/// A finished room in 3D: drag to look around, pinch to walk forward and back,
/// two fingers to step sideways. Opens at the pipeline's starting camera.
struct SplatScreen: View {
    let splat: URL
    let view: URL?
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = SplatModel()

    var body: some View {
        ZStack(alignment: .top) {
            SplatMetalView(model: model)
                .ignoresSafeArea()
                .gesture(DragGesture(minimumDistance: 0).onChanged { model.drag($0.translation) }.onEnded { _ in model.endDrag() })
                .simultaneousGesture(MagnifyGesture().onChanged { model.pinch($0.magnification) }.onEnded { _ in model.endPinch() })
            HStack {
                Button { dismiss() } label: {
                    Image(systemName: "xmark").font(.headline).frame(width: 40, height: 40)
                        .background(.ultraThinMaterial, in: Circle())
                }
                Spacer()
                if let error = model.error {
                    Text(error).font(.footnote).padding(8).background(.red.opacity(0.8), in: Capsule())
                } else if model.loading {
                    Label("Loading \(model.loaded) splats", systemImage: "hourglass")
                        .font(.footnote).padding(8).background(.ultraThinMaterial, in: Capsule())
                }
                Spacer()
                Button { model.resetCamera() } label: {
                    Image(systemName: "scope").font(.headline).frame(width: 40, height: 40)
                        .background(.ultraThinMaterial, in: Circle())
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)
            VStack {
                Spacer()
                Text("Drag to look · pinch to walk")
                    .font(.caption)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(.bottom, 24)
            }
        }
        .task { await model.load(splat: splat, view: view) }
    }
}

@MainActor
final class SplatModel: ObservableObject {
    @Published var loading = true
    @Published var loaded = 0
    @Published var error: String?

    let device = MTLCreateSystemDefaultDevice()!
    private(set) var renderer: SplatRenderer?
    var camera = TouchCamera()
    private var lastDrag = CGSize.zero
    private var lastPinch: CGFloat = 1

    func load(splat: URL, view: URL?) async {
        do {
            let reader = try AutodetectSceneReader(splat)
            var points: [SplatPoint] = []
            for try await batch in try await reader.read() {
                points.append(contentsOf: batch)
                loaded = points.count
            }
            guard !points.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
            let renderer = try SplatRenderer(device: device, colorFormat: .bgra8Unorm_srgb, depthFormat: .depth32Float,
                                             sampleCount: 1, maxViewCount: 1, maxSimultaneousRenders: 3,
                                             highQualityDepth: true,
                                             clearColor: MTLClearColor(red: 0.035, green: 0.04, blue: 0.04, alpha: 1))
            await renderer.addChunk(try SplatChunk(device: device, from: points))
            camera.configure(points: points, startView: view)
            self.renderer = renderer
            loading = false
        } catch {
            self.error = "Could not open this splat"
            loading = false
        }
    }

    func drag(_ translation: CGSize) {
        let dx = Float(translation.width - lastDrag.width), dy = Float(translation.height - lastDrag.height)
        lastDrag = translation
        camera.look(yaw: -dx * 0.005, pitch: -dy * 0.005)
    }

    func endDrag() { lastDrag = .zero }

    func pinch(_ magnification: CGFloat) {
        let step = Float(magnification - lastPinch)
        lastPinch = magnification
        camera.walk(step * 2)
    }

    func endPinch() { lastPinch = 1 }

    func resetCamera() { camera.reset() }
}

/// Look-around camera for touch, in the splat's own frame.
struct TouchCamera {
    private var up = SIMD3<Float>(0, -1, 0)
    private var forward0 = SIMD3<Float>(0, 0, 1)
    private var startPosition = SIMD3<Float>.zero
    private var startPitch: Float = 0
    private(set) var position = SIMD3<Float>.zero
    private var yaw: Float = 0
    private var pitch: Float = 0
    private var fovY: Float = 55 * .pi / 180
    private var metre: Float = 1
    private var near: Float = 0.01
    private var far: Float = 1000

    /// From <splat>.view.json (world-to-camera, column-major, x right, y down,
    /// z forward) when there is one, else from outside the splat's middle.
    mutating func configure(points: [SplatPoint], startView: URL?) {
        let stride = max(1, points.count / 20_000)
        let sample = Swift.stride(from: 0, to: points.count, by: stride).map { points[$0].position }
        let centre = sample.reduce(SIMD3<Float>.zero, +) / Float(max(sample.count, 1))
        let extent = sample.map { simd_distance($0, centre) }.sorted().dropLast(sample.count / 20).last ?? 1
        near = extent * 0.002
        far = extent * 20
        if let startView, let data = try? Data(contentsOf: startView),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let values = json["viewMatrix"] as? [Double], values.count == 16 {
            let m = values.map { Float($0) }
            let right = SIMD3(m[0], m[4], m[8]), down = SIMD3(m[1], m[5], m[9]), forward = SIMD3(m[2], m[6], m[10])
            let t = SIMD3(m[12], m[13], m[14])
            startPosition = -(right * t.x + down * t.y + forward * t.z)
            up = -down
            if let u = json["up"] as? [Double], u.count == 3 { up = simd_normalize(SIMD3(Float(u[0]), Float(u[1]), Float(u[2]))) }
            if let fov = json["fovY"] as? Double { fovY = Float(fov) * .pi / 180 }
            if let m = json["metre"] as? Double, m > 0 {
                metre = Float(m)
                near = metre * 0.02
            }
            var level = forward - up * simd_dot(forward, up)
            if simd_length(level) < 1e-4 { level = SIMD3(1, 0, 0) }
            forward0 = simd_normalize(level)
            startPitch = asin(min(max(simd_dot(simd_normalize(forward), up), -1), 1))
        } else {
            metre = extent / 3
            startPosition = centre - forward0 * extent * 0.8
        }
        reset()
    }

    mutating func reset() {
        position = startPosition
        yaw = 0
        pitch = startPitch
    }

    mutating func look(yaw dYaw: Float, pitch dPitch: Float) {
        yaw += dYaw
        pitch = min(max(pitch + dPitch, -1.4), 1.4)
    }

    mutating func walk(_ amount: Float) {
        position += lookDirection * amount * metre
    }

    private var heading: SIMD3<Float> {
        cos(yaw) * forward0 + sin(yaw) * simd_cross(up, forward0)
    }

    private var lookDirection: SIMD3<Float> {
        cos(pitch) * heading + sin(pitch) * up
    }

    var viewMatrix: simd_float4x4 {
        let f = lookDirection
        let r = simd_normalize(simd_cross(f, up))
        let u = simd_cross(r, f)
        let p = position
        return simd_float4x4(columns: (SIMD4(r.x, u.x, -f.x, 0), SIMD4(r.y, u.y, -f.y, 0), SIMD4(r.z, u.z, -f.z, 0),
                                      SIMD4(-simd_dot(r, p), -simd_dot(u, p), simd_dot(f, p), 1)))
    }

    func projection(aspect: Float) -> simd_float4x4 {
        let ys = 1 / tan(fovY / 2), xs = ys / aspect, zs = far / (near - far)
        return simd_float4x4(columns: (SIMD4(xs, 0, 0, 0), SIMD4(0, ys, 0, 0), SIMD4(0, 0, zs, -1), SIMD4(0, 0, zs * near, 0)))
    }
}

struct SplatMetalView: UIViewRepresentable {
    let model: SplatModel

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: model.device)
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.preferredFramesPerSecond = 60
        view.delegate = context.coordinator
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}

    final class Coordinator: NSObject, MTKViewDelegate {
        let model: SplatModel
        private let queue: MTLCommandQueue

        init(model: SplatModel) {
            self.model = model
            queue = model.device.makeCommandQueue()!
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            MainActor.assumeIsolated {
                guard let renderer = model.renderer, let drawable = view.currentDrawable,
                      let buffer = queue.makeCommandBuffer() else { return }
                let width = Int(view.drawableSize.width), height = Int(view.drawableSize.height)
                let viewport = SplatRenderer.ViewportDescriptor(
                    viewport: MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: 0, zfar: 1),
                    projectionMatrix: model.camera.projection(aspect: Float(width) / Float(max(height, 1))),
                    viewMatrix: model.camera.viewMatrix,
                    screenSize: SIMD2(x: width, y: height))
                let drew = (try? renderer.render(viewports: [viewport], colorTexture: drawable.texture, colorStoreAction: .store,
                                                 depthTexture: view.depthStencilTexture, rasterizationRateMap: nil,
                                                 renderTargetArrayLength: 0, accessTimeout: 0.05, sortTimeout: 0.05,
                                                 to: buffer)) ?? false
                if drew { buffer.present(drawable) }
                buffer.commit()
            }
        }
    }
}
