import AppKit
import Metal
import SwiftUI

/// `Splat Viewer --selftest <output folder> <splat file>`: renders the landing page and
/// the splat from the starting camera and after simulated key presses into PNGs, writes
/// selftest.txt, and quits. It checks the screens and the camera without a person at
/// the keyboard.
@MainActor
enum SelfTest {
    static func run(outputDirectory: URL, splat: URL) {
        try? FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        var log: [String] = []
        let folder = splat.deletingLastPathComponent()
        log.append("splat check: \(SplatFile.problem(with: splat) ?? "ok")")
        log.append("cloud-dense.ply check: \(SplatFile.problem(with: folder.appendingPathComponent("cloud-dense.ply")) ?? "ok")")
        log.append("folder of spaces expands to: \(SplatFile.expand(folder.deletingLastPathComponent()).map { "\($0.deletingLastPathComponent().lastPathComponent)/\($0.lastPathComponent)" })")

        // The landing page: empty, with splats, and while a file is dragged over it.
        let spaces = SplatFile.expand(folder.deletingLastPathComponent())
        let full = Library.scratch(with: spaces.isEmpty ? [splat] : Array(spaces.prefix(6)))
        full.openTab(full.items[0].id, select: false)
        full.notice = "cloud-dense.ply is a point cloud, not a Gaussian splat"
        snapshotView(HomeContent(dropTargeted: false).environment(full).tint(.oasis)
                        .background(Color(nsColor: .windowBackgroundColor)), name: "home",
                     size: CGSize(width: 1280, height: 1250), into: outputDirectory)
        snapshotView(HomeContent(dropTargeted: true).environment(Library.scratch(with: [])).tint(.oasis)
                        .background(Color(nsColor: .windowBackgroundColor)),
                     name: "home-empty-dragging", size: CGSize(width: 1280, height: 820), into: outputDirectory)
        snapshotView(ControlsPanel(style: .overlay).environment(\.colorScheme, .dark).padding(20)
                        .background(Color(red: 0.035, green: 0.04, blue: 0.04)),
                     name: "viewer-controls", size: CGSize(width: 360, height: 400), into: outputDirectory)
        full.notice = nil
        for item in full.items.prefix(3) { full.openTab(item.id, select: false) }
        full.selected = full.openTabs.last
        snapshotView(TabStrip().environment(full), name: "tabs", size: CGSize(width: 1100, height: 40),
                     into: outputDirectory)
        log.append("tabs: \(full.openTabs.count) open; ⌘W closes the selected: \(full.closeSelected()), now \(full.openTabs.count)")

        let lines = log
        Task.detached(priority: .userInitiated) {
            var log = lines
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                log.append("no Metal device")
                await finish(log, outputDirectory)
                return
            }
            do {
                let started = Date()
                let scene = try await SplatScene.load(url: splat, device: device) { _ in }
                log.append(String(format: "loaded %d splats in %.1f s; start view: %@", scene.count,
                                  Date().timeIntervalSince(started), scene.start?.frame ?? "none"))
                var camera = FlyCamera()
                camera.configure(bounds: scene.bounds, start: scene.start)
                func shot(_ name: String) {
                    let image = scene.snapshot(camera: camera, width: 960, height: 600, commandQueue: queue)
                    if let png = image?.pngData() {
                        try? png.write(to: outputDirectory.appendingPathComponent("\(name).png"))
                    }
                    let p = camera.pose
                    log.append(String(format: "%@: %@  position (%.2f, %.2f, %.2f) yaw %.0f° pitch %.0f°",
                                      name, image == nil ? "render FAILED" : "rendered",
                                      p.position.x, p.position.y, p.position.z,
                                      p.yaw * 180 / .pi, p.pitch * 180 / .pi))
                }
                func hold(_ keys: Set<MoveKey>, _ flags: NSEvent.ModifierFlags, seconds: Float) {
                    let steps = Int(seconds * 60)
                    let input = FlyCamera.Input(held: keys, flags: flags)
                    for _ in 0..<steps { camera.advance(dt: 1 / 60, input: input) }
                    for _ in 0..<60 { camera.advance(dt: 1 / 60, input: FlyCamera.Input()) }   // let go
                }
                shot("1-start")
                hold([.up], [], seconds: 1)
                shot("2-arrow-up-1s-forward")
                hold([.left], [.command], seconds: 1)
                shot("3-cmd-left-1s-turn-left")
                hold([.up], [.command], seconds: 0.5)
                shot("4-cmd-up-half-s-look-up")
                hold([.right], [], seconds: 1)
                shot("5-arrow-right-1s-step-right")
                camera.reset()
                shot("6-reset")
            } catch {
                log.append("load failed: \(error.localizedDescription)")
            }
            await finish(log, outputDirectory)
        }
    }

    /// `--keytest <splat> <log file>`: opens the splat in the real window, posts arrow and
    /// ⌘ + arrow key events through the app's event loop, and logs where the camera went
    /// and whether every key was let go.
    static func runKeyTest(splat: URL, log logURL: URL) {
        SceneController.testMode = true
        let library = Library.shared
        library.add([splat])
        Task { @MainActor in
            var log: [String] = []
            @MainActor func controller() -> SceneController? {
                SceneController.instances.allObjects.first { $0.item.path == splat.standardizedFileURL.path && $0.isReady }
            }
            for _ in 0..<200 where controller() == nil { try? await Task.sleep(for: .milliseconds(100)) }
            guard let scene = controller(),
                  let window = NSApp.windows.first(where: { $0.isVisible && !($0 is NSPanel) }) else {
                let found = SceneController.instances.allObjects.map { "\($0.item.path) ready=\($0.isReady)" }
                let windows = NSApp.windows.map { "\(type(of: $0)) visible=\($0.isVisible)" }
                try? "not ready\ncontrollers: \(found)\nwindows: \(windows)\ntabs: \(library.openTabs.count) selected: \(String(describing: library.selected))\n"
                    .write(to: logURL, atomically: true, encoding: .utf8)
                NSApp.terminate(nil)
                return
            }
            func post(_ type: NSEvent.EventType, _ keyCode: UInt16, _ flags: NSEvent.ModifierFlags = [], _ chars: String = "") {
                if let event = NSEvent.keyEvent(with: type, location: .zero, modifierFlags: flags,
                                                timestamp: ProcessInfo.processInfo.systemUptime,
                                                windowNumber: window.windowNumber, context: nil,
                                                characters: chars, charactersIgnoringModifiers: chars,
                                                isARepeat: false, keyCode: keyCode) {
                    NSApp.postEvent(event, atStart: false)
                }
            }
            @MainActor func describe(_ label: String) {
                let p = scene.debugPose
                log.append(String(format: "%@: position (%.2f, %.2f, %.2f) yaw %.0f° pitch %.0f° held %@",
                                  label, p.position.x, p.position.y, p.position.z,
                                  p.yaw * 180 / .pi, p.pitch * 180 / .pi, "\(scene.debugHeldKeys.count)"))
            }
            describe("start")
            post(.keyDown, 126)                              // ↑
            try? await Task.sleep(for: .seconds(1))
            post(.keyUp, 126)
            try? await Task.sleep(for: .milliseconds(700))
            describe("after ↑ for 1 s")
            post(.flagsChanged, 55, .command)                // ⌘ down
            post(.keyDown, 123, .command)                    // ⌘←
            try? await Task.sleep(for: .seconds(1))
            post(.keyUp, 123, .command)                      // ← up while ⌘ is still held
            try? await Task.sleep(for: .milliseconds(700))
            describe("after ⌘← for 1 s")
            post(.keyDown, 126, .command)                    // ⌘↑
            try? await Task.sleep(for: .milliseconds(500))
            post(.keyUp, 126, .command)
            post(.flagsChanged, 55, [])                      // ⌘ up
            try? await Task.sleep(for: .milliseconds(700))
            describe("after ⌘↑ for 0.5 s")
            post(.keyDown, 15, [], "r")                      // R
            try? await Task.sleep(for: .milliseconds(300))
            describe("after R")
            let before = library.openTabs.count
            post(.keyDown, 13, .command, "w")                // ⌘W
            try? await Task.sleep(for: .milliseconds(300))
            log.append("⌘W: open tabs \(before) -> \(library.openTabs.count), showing \(library.selected == nil ? "the library" : "a splat")")
            try? log.joined(separator: "\n").appending("\n").write(to: logURL, atomically: true, encoding: .utf8)
            NSApp.terminate(nil)
        }
    }

    /// Writes how many windows and tabs are open five seconds after launch, then quits:
    /// checks that opening a file on a cold launch shows the window.
    static func reportWindows(to logURL: URL) {
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(5))
            let windows = NSApp.windows.filter { $0.isVisible && !($0 is NSPanel) }.count
            let library = Library.shared
            try? "windows: \(windows)\ntabs: \(library.openTabs.count)\nselected: \(library.selected.flatMap { library.item($0)?.title } ?? "library")\n"
                .write(to: logURL, atomically: true, encoding: .utf8)
            NSApp.terminate(nil)
        }
    }

    private static func snapshotView<V: View>(_ view: V, name: String, size: CGSize = CGSize(width: 1280, height: 900),
                                              into directory: URL) {
        let renderer = ImageRenderer(content: view.frame(width: size.width, height: size.height))
        renderer.scale = 1
        if let image = renderer.cgImage, let png = image.pngData() {
            try? png.write(to: directory.appendingPathComponent("\(name).png"))
        }
    }

    private static func finish(_ log: [String], _ directory: URL) {
        try? log.joined(separator: "\n").appending("\n")
            .write(to: directory.appendingPathComponent("selftest.txt"), atomically: true, encoding: .utf8)
        NSApp.terminate(nil)
    }
}
