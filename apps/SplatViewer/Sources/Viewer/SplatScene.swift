import AppKit
import Metal
import MetalSplatter
import simd
import SplatIO

enum SceneError: LocalizedError {
    case empty

    var errorDescription: String? {
        switch self {
        case .empty: "The file holds no splats."
        }
    }
}

/// A loaded splat on the GPU, ready to draw from any camera.
final class SplatScene: @unchecked Sendable {
    static let colorFormat = MTLPixelFormat.bgra8Unorm_srgb
    static let depthFormat = MTLPixelFormat.depth32Float
    static let clearColor = MTLClearColor(red: 0.035, green: 0.04, blue: 0.04, alpha: 1)

    let device: MTLDevice
    let renderer: SplatRenderer
    let count: Int
    let bounds: SceneBounds
    let start: StartView?

    private init(device: MTLDevice, renderer: SplatRenderer, count: Int, bounds: SceneBounds, start: StartView?) {
        self.device = device
        self.renderer = renderer
        self.count = count
        self.bounds = bounds
        self.start = start
    }

    /// Reads a .ply, .splat or .spz file and uploads it. `progress` gets the number of
    /// splats read so far.
    static func load(url: URL, device: MTLDevice,
                     progress: @escaping @Sendable (Int) -> Void) async throws -> SplatScene {
        let reader = try AutodetectSceneReader(url)
        var points: [SplatPoint] = []
        if let expected = SplatFile.expectedCount(url) { points.reserveCapacity(expected) }
        var reported = 0
        for try await batch in try await reader.read() {
            try Task.checkCancellation()
            points.append(contentsOf: batch)
            if points.count - reported >= 20_000 {
                reported = points.count
                progress(reported)
            }
        }
        guard !points.isEmpty else { throw SceneError.empty }
        progress(points.count)
        let stride = max(1, points.count / 60_000)
        let bounds = SceneBounds(positions: Swift.stride(from: 0, to: points.count, by: stride)
            .map { points[$0].position })
        let count = points.count
        let chunk = try SplatChunk(device: device, from: points)
        points = []
        try Task.checkCancellation()
        let renderer = try SplatRenderer(device: device,
                                         colorFormat: colorFormat,
                                         depthFormat: depthFormat,
                                         sampleCount: 1,
                                         maxViewCount: 1,
                                         maxSimultaneousRenders: 3,
                                         highQualityDepth: true,
                                         clearColor: clearColor)
        await renderer.addChunk(chunk)
        return SplatScene(device: device, renderer: renderer, count: count, bounds: bounds,
                          start: StartView.load(besides: url))
    }

    func viewport(width: Int, height: Int, camera: FlyCamera,
                  pose: FlyCamera.Pose? = nil) -> SplatRenderer.ViewportDescriptor {
        SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height),
                                  znear: 0, zfar: 1),
            projectionMatrix: camera.projection(aspect: Float(width) / Float(max(height, 1))),
            viewMatrix: camera.viewMatrix(pose),
            screenSize: SIMD2(x: width, y: height))
    }

    /// Encodes one frame; false when the renderer is not ready and the frame should be dropped.
    @discardableResult
    func render(camera: FlyCamera, pose: FlyCamera.Pose? = nil, width: Int, height: Int,
                colorTexture: MTLTexture, depthTexture: MTLTexture?,
                commandBuffer: MTLCommandBuffer, timeout: TimeInterval = 0.05) throws -> Bool {
        try renderer.render(viewports: [viewport(width: width, height: height, camera: camera, pose: pose)],
                            colorTexture: colorTexture,
                            colorStoreAction: .store,
                            depthTexture: depthTexture,
                            rasterizationRateMap: nil,
                            renderTargetArrayLength: 0,
                            accessTimeout: timeout,
                            sortTimeout: timeout,
                            to: commandBuffer)
    }

    /// Renders one image offscreen, for thumbnails and the self-test. Blocks until done:
    /// call it off the main thread. It renders twice with a pause between, so the splats
    /// are sorted for this camera rather than the last one.
    func snapshot(camera: FlyCamera, pose: FlyCamera.Pose? = nil, width: Int, height: Int,
                  commandQueue: MTLCommandQueue) -> CGImage? {
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: Self.colorFormat, width: width, height: height, mipmapped: false)
        colorDescriptor.usage = [.renderTarget, .shaderRead]
        colorDescriptor.storageMode = .shared
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: Self.depthFormat, width: width, height: height, mipmapped: false)
        depthDescriptor.usage = [.renderTarget]
        depthDescriptor.storageMode = .private
        guard let color = device.makeTexture(descriptor: colorDescriptor),
              let depth = device.makeTexture(descriptor: depthDescriptor) else { return nil }

        var rendered = 0
        for _ in 0..<60 {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }
            let ok = (try? render(camera: camera, pose: pose, width: width, height: height,
                                  colorTexture: color, depthTexture: depth,
                                  commandBuffer: commandBuffer, timeout: 0.5)) ?? false
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if ok {
                rendered += 1
                if rendered == 2 { break }
            }
            Thread.sleep(forTimeInterval: 0.3)
        }
        guard rendered == 2 else { return nil }

        let bytesPerRow = width * 4
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * height)
        color.getBytes(&bytes, bytesPerRow: bytesPerRow,
                       from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow, space: space,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                          | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        return context.makeImage()
    }
}

extension CGImage {
    func pngData() -> Data? {
        NSBitmapImageRep(cgImage: self).representation(using: .png, properties: [:])
    }
}
