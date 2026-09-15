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

/// A loaded splat on the GPU, ready to draw from any camera. When the pipeline's room
/// model sits beside it, the splat is split into pieces the editor can change on their
/// own: one per detected object, one per wall, and the rest.
final class SplatScene: @unchecked Sendable {
    static let colorFormat = MTLPixelFormat.bgra8Unorm_srgb
    static let depthFormat = MTLPixelFormat.depth32Float
    static let clearColor = MTLClearColor(red: 0.035, green: 0.04, blue: 0.04, alpha: 1)

    let device: MTLDevice
    let renderer: SplatRenderer
    let count: Int
    let bounds: SceneBounds
    let start: StartView?
    let room: RoomModel?
    let grid: OccupancyGrid?

    /// A piece of the splat that edits change, with its original Gaussians.
    private struct Part {
        var points: [SplatPoint]
        var chunk: ChunkID?
        var placement = Placement()
        var colour: String?
        var removed = false
        var reference: GaussianMath.PaintReference?
    }
    /// What covers the floor and walls an object had hidden, shown once it moves or goes.
    private struct Patch {
        var floor: [SplatPoint] = []
        var walls: [String: [SplatPoint]] = [:]
        var floorChunk: ChunkID?
        var wallChunks: [String: (chunk: ChunkID, colour: String?)] = [:]
        var shown = true
    }
    private var objectParts: [String: Part] = [:]
    private var patches: [String: Patch] = [:]
    /// Gaussians on the floor outside every object, which patches continue.
    private var floorPoints: [SplatPoint] = []
    private var wallParts: [String: Part] = [:]
    private var addedParts: [UUID: (item: AddedItem, chunk: ChunkID)] = [:]

    private init(device: MTLDevice, renderer: SplatRenderer, count: Int, bounds: SceneBounds,
                 start: StartView?, room: RoomModel?, grid: OccupancyGrid?) {
        self.device = device
        self.renderer = renderer
        self.count = count
        self.bounds = bounds
        self.start = start
        self.room = room
        self.grid = grid
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
        let start = StartView.load(besides: url)
        let room = RoomModel.load(besides: url)
        let metre = room?.metre ?? start?.metre.map { Float($0) }
        let grid = OccupancyGrid(positions: points.filter { $0.opacity.asLinearFloat > 0.3 }.map(\.position),
                                 cell: metre.map { $0 * 0.12 } ?? bounds.extent / 80)
        let renderer = try SplatRenderer(device: device,
                                         colorFormat: colorFormat,
                                         depthFormat: depthFormat,
                                         sampleCount: 1,
                                         maxViewCount: 1,
                                         maxSimultaneousRenders: 3,
                                         highQualityDepth: true,
                                         clearColor: clearColor)
        let scene = SplatScene(device: device, renderer: renderer, count: points.count, bounds: bounds,
                               start: start, room: room, grid: grid)
        try Task.checkCancellation()
        if let room {
            try await scene.split(points, room: room)
        } else {
            await renderer.addChunk(try SplatChunk(device: device, from: points))
        }
        return scene
    }

    /// Sorts every Gaussian into the object box or wall it belongs to, or the rest.
    private func split(_ points: [SplatPoint], room: RoomModel) async throws {
        let m = room.metre
        // Smaller objects first, so a lamp on a desk is the lamp's.
        let objects = room.objects.sorted { $0.size.x * $0.size.y * $0.size.z < $1.size.x * $1.size.y * $1.size.z }
        var objectPoints = [[SplatPoint]](repeating: [], count: objects.count)
        var wallPoints = [[SplatPoint]](repeating: [], count: room.walls.count)
        var rest: [SplatPoint] = []
        rest.reserveCapacity(points.count)
        let margin = 0.05 * m, wallBand = 0.08 * m
        let floorTop = room.floorZ + 0.02 * m, ceiling = room.floorZ + room.height
        for point in points {
            let p = room.toScene(point.position)
            let inHeight = p.z > room.floorZ + 0.03 * m && p.z < ceiling
            let nearWall = inHeight ? room.walls.firstIndex { (wall: RoomModel.Wall) -> Bool in
                let d: SIMD3<Float> = p - wall.centre
                let across: Float = abs(simd_dot(d, wall.inward))
                let along: Float = abs(simd_dot(d, wall.along))
                return across < wallBand && along < wall.halfLength + 0.1 * m
            } : nil
            let inObject = p.z > floorTop ? objects.firstIndex { (object: RoomModel.Object) -> Bool in
                let lo: SIMD3<Float> = object.min - margin
                let hi: SIMD3<Float> = object.max + margin
                return p.x > lo.x && p.x < hi.x && p.y > lo.y && p.y < hi.y && p.z < hi.z
            } : nil
            // Paint within 5 cm of a wall stays with the wall, even behind furniture.
            let wallDistance = nearWall.map { abs(simd_dot(p - room.walls[$0].centre, room.walls[$0].inward)) }
            if let index = inObject, (wallDistance ?? .infinity) > 0.05 * m {
                objectPoints[index].append(point)
            } else if let wall = nearWall {
                wallPoints[wall].append(point)
            } else {
                rest.append(point)
            }
        }
        floorPoints = rest.filter { abs(room.toScene($0.position).z - room.floorZ) < 0.05 * m }
        if !rest.isEmpty { await renderer.addChunk(try SplatChunk(device: device, from: rest)) }
        for (index, object) in objects.enumerated() where !objectPoints[index].isEmpty {
            let chunk = await renderer.addChunk(try SplatChunk(device: device, from: objectPoints[index]))
            objectParts[object.id] = Part(points: objectPoints[index], chunk: chunk)
        }
        for (index, wall) in room.walls.enumerated() where !wallPoints[index].isEmpty {
            let chunk = await renderer.addChunk(try SplatChunk(device: device, from: wallPoints[index]))
            wallParts[wall.id] = Part(points: wallPoints[index], chunk: chunk)
        }
    }

    // MARK: Edits

    /// Brings the GPU pieces in line with `edits`: rebuilds what moved, was resized or
    /// repainted, hides what was removed, and builds what was added. Call from one task at
    /// a time; the heavy work runs off the main thread.
    func apply(_ edits: SceneEdits) async {
        guard let room else { return }
        let device = self.device
        for object in room.objects {
            guard var part = objectParts[object.id] else { continue }
            let placement = edits.placement(object.id)
            let removed = edits.removed.contains(object.id)
            if placement != part.placement {
                let source = part.points
                let chunk = await Task.detached(priority: .userInitiated) { () -> SplatChunk? in
                    let moved = placement.isIdentity ? source
                        : GaussianMath.place(source, object: object, placement: placement, room: room)
                    return try? SplatChunk(device: device, from: moved)
                }.value
                if let chunk {
                    let id = await renderer.addChunk(chunk, sortByLocality: false, enabled: !removed)
                    if let old = part.chunk { await renderer.removeChunk(old) }
                    part.chunk = id
                    part.removed = removed
                }
                part.placement = placement
            }
            if removed != part.removed, let chunk = part.chunk {
                await renderer.setChunkEnabled(chunk, enabled: !removed)
                part.removed = removed
            }
            objectParts[object.id] = part
            await updatePatch(for: object, show: removed || !placement.isIdentity, edits: edits, room: room)
        }
        for wall in room.walls {
            guard var part = wallParts[wall.id] else { continue }
            let colour = edits.wallColours[wall.id]
            guard colour != part.colour else { continue }
            let source = part.points
            let known = part.reference
            let result = await Task.detached(priority: .userInitiated) { () -> (SplatChunk?, GaussianMath.PaintReference) in
                let reference = known ?? GaussianMath.paintReference(source, wall: wall, room: room)
                let painted = colour.flatMap { SIMD3<Float>(hex: $0) }
                    .map { GaussianMath.paint(source, colour: $0, reference: reference, wall: wall, room: room) } ?? source
                return (try? SplatChunk(device: device, from: painted), reference)
            }.value
            part.reference = result.1
            if let chunk = result.0 {
                let id = await renderer.addChunk(chunk, sortByLocality: false)
                if let old = part.chunk { await renderer.removeChunk(old) }
                part.chunk = id
            }
            part.colour = colour
            wallParts[wall.id] = part
        }
        let wanted = Dictionary(uniqueKeysWithValues: edits.added.map { ($0.id, $0) })
        for (id, entry) in addedParts where wanted[id] == nil {
            await renderer.removeChunk(entry.chunk)
            addedParts[id] = nil
        }
        for item in edits.added where addedParts[item.id]?.item != item {
            let chunk = await Task.detached(priority: .userInitiated) { () -> SplatChunk? in
                try? SplatChunk(device: device, from: FurnitureBuilder.points(for: item, room: room))
            }.value
            guard let chunk else { continue }
            let id = await renderer.addChunk(chunk, sortByLocality: false)
            if let old = addedParts[item.id]?.chunk { await renderer.removeChunk(old) }
            addedParts[item.id] = (item, id)
        }
    }

    /// Builds an object's patch the first time it is needed, keeps its wall pieces painted
    /// like their walls, and shows or hides it.
    private func updatePatch(for object: RoomModel.Object, show: Bool, edits: SceneEdits, room: RoomModel) async {
        let device = self.device
        if patches[object.id] == nil {
            guard show else { return }
            let floor = floorPoints
            let walls = wallParts.mapValues(\.points)
            let built = await Task.detached(priority: .userInitiated) {
                SurfacePatch.patches(for: object, room: room, floor: floor, walls: walls)
            }.value
            var patch = Patch(floor: built.floor, walls: built.walls)
            if !built.floor.isEmpty, let chunk = try? SplatChunk(device: device, from: built.floor) {
                patch.floorChunk = await renderer.addChunk(chunk, sortByLocality: false)
            }
            patches[object.id] = patch
        }
        guard var patch = patches[object.id] else { return }
        for (wallID, points) in patch.walls {
            let colour = edits.wallColours[wallID]
            if let existing = patch.wallChunks[wallID], existing.colour == colour { continue }
            guard let wall = room.wall(wallID) else { continue }
            var reference = wallParts[wallID]?.reference
            let source = wallParts[wallID]?.points ?? []
            let result = await Task.detached(priority: .userInitiated) { () -> (SplatChunk?, GaussianMath.PaintReference?) in
                guard let hex = colour, let rgb = SIMD3<Float>(hex: hex) else {
                    return (try? SplatChunk(device: device, from: points), reference)
                }
                let known = reference ?? GaussianMath.paintReference(source, wall: wall, room: room)
                let painted = GaussianMath.paint(points, colour: rgb, reference: known, wall: nil, room: room)
                return (try? SplatChunk(device: device, from: painted), known)
            }.value
            reference = result.1
            if let reference { wallParts[wallID]?.reference = reference }
            guard let chunk = result.0 else { continue }
            let id = await renderer.addChunk(chunk, sortByLocality: false, enabled: patch.shown)
            if let old = patch.wallChunks[wallID]?.chunk { await renderer.removeChunk(old) }
            patch.wallChunks[wallID] = (id, colour)
        }
        if patch.shown != show {
            patch.shown = show
            if let chunk = patch.floorChunk { await renderer.setChunkEnabled(chunk, enabled: show) }
            for entry in patch.wallChunks.values { await renderer.setChunkEnabled(entry.chunk, enabled: show) }
        }
        patches[object.id] = patch
    }

    /// Test hook: what painting a wall would pick out.
    func debugPaint(_ wallID: String) -> String {
        guard let room, let wall = room.wall(wallID), let part = wallParts[wallID] else { return "\(wallID): no part" }
        let reference = GaussianMath.paintReference(part.points, wall: wall, room: room)
        let painted = GaussianMath.paint(part.points, colour: SIMD3(0.7, 0.76, 0.65), reference: reference, wall: wall, room: room)
        let changed = zip(part.points, painted).filter { simd_length($0.color.asSRGBFloat - $1.color.asSRGBFloat) > 0.05 }.count
        let distances = part.points.map { abs(simd_dot(room.toScene($0.position) - wall.centre, wall.inward)) / room.metre }
        let near = [0.025, 0.045, 0.08].map { limit in distances.filter { $0 < Float(limit) }.count }
        return String(format: "%@: %d points, within 2.5/4.5/8 cm %@; reference %@ lum %.2f; changed %d",
                      wallID, part.points.count, "\(near)", "\(reference.colour)", reference.luminance, changed)
    }

    // MARK: Drawing

    func viewport(width: Int, height: Int, camera: FlyCamera, pose: FlyCamera.Pose? = nil,
                  near: Float? = nil) -> SplatRenderer.ViewportDescriptor {
        SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height),
                                  znear: 0, zfar: 1),
            projectionMatrix: camera.projection(aspect: Float(width) / Float(max(height, 1)), near: near),
            viewMatrix: camera.viewMatrix(pose),
            screenSize: SIMD2(x: width, y: height))
    }

    /// Encodes one frame; false when the renderer is not ready and the frame should be dropped.
    /// `near` pushes the near clipping plane out, to look past something close by.
    @discardableResult
    func render(camera: FlyCamera, pose: FlyCamera.Pose? = nil, near: Float? = nil, width: Int, height: Int,
                colorTexture: MTLTexture, depthTexture: MTLTexture?,
                commandBuffer: MTLCommandBuffer, timeout: TimeInterval = 0.05) throws -> Bool {
        try renderer.render(viewports: [viewport(width: width, height: height, camera: camera, pose: pose, near: near)],
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
    func snapshot(camera: FlyCamera, pose: FlyCamera.Pose? = nil, near: Float? = nil, width: Int, height: Int,
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
            let ok = (try? render(camera: camera, pose: pose, near: near, width: width, height: height,
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
