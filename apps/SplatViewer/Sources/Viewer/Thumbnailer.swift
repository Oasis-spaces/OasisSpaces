import Foundation
import Metal

/// Makes a thumbnail for every library item that lacks one, one splat at a time in the
/// background, so the landing page shows what each splat is before it is opened.
@MainActor
final class Thumbnailer {
    static let shared = Thumbnailer()

    private var queued: [UUID] = []
    private var running = false
    private var failed: Set<UUID> = []

    func refresh(_ library: Library) {
        for item in library.items
        where item.exists && !failed.contains(item.id) && !queued.contains(item.id)
            && !FileManager.default.fileExists(atPath: library.thumbnailURL(for: item.id).path) {
            queued.append(item.id)
        }
        runNext(library)
    }

    private func runNext(_ library: Library) {
        guard !running, !queued.isEmpty else { return }
        let id = queued.removeFirst()
        guard let item = library.item(id), item.exists,
              !FileManager.default.fileExists(atPath: library.thumbnailURL(for: id).path) else {
            runNext(library)
            return
        }
        running = true
        let url = item.url
        let output = library.thumbnailURL(for: id)
        Task.detached(priority: .utility) {
            var saved = false
            if let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
               let scene = try? await SplatScene.load(url: url, device: device, progress: { _ in }) {
                var camera = FlyCamera()
                camera.configure(bounds: scene.bounds, start: scene.start)
                if let png = scene.snapshot(camera: camera, width: 640, height: 400, commandQueue: queue)?.pngData() {
                    saved = (try? png.write(to: output, options: .atomic)) != nil
                }
            }
            await MainActor.run {
                self.running = false
                if saved { library.thumbnailSaved(id) } else { self.failed.insert(id) }
                self.runNext(library)
            }
        }
    }
}
