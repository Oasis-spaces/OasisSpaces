import AppKit
import MetalKit
import SwiftUI

/// What the SwiftUI overlay shows about one open splat.
@MainActor
@Observable
final class ViewerModel {
    enum Phase: Equatable {
        case loading(read: Int, total: Int?)
        case ready
        case failed(String)
    }

    var phase: Phase = .loading(read: 0, total: nil)
    var splatCount = 0
    var startViewFrame: String?
    var showHelp = true
    var helpAutoHidden = false
    let editor = EditorModel()
    @ObservationIgnored weak var controller: SceneController?
}

struct SplatSceneView: NSViewRepresentable {
    let item: SplatItem
    let isActive: Bool
    let model: ViewerModel
    let library: Library

    func makeCoordinator() -> SceneController {
        SceneController(item: item, model: model, library: library)
    }

    func makeNSView(context: Context) -> SplatMTKView {
        let view = SplatMTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        context.coordinator.attach(view)
        context.coordinator.setActive(isActive)
        return view
    }

    func updateNSView(_ view: SplatMTKView, context: Context) {
        context.coordinator.setActive(isActive)
    }

    static func dismantleNSView(_ view: SplatMTKView, coordinator: SceneController) {
        coordinator.detach()
    }
}

/// The Metal view: forwards the mouse and trackpad to its controller. Keys are read by
/// the controller's event monitor, which also sees ⌘ + arrow key-ups.
final class SplatMTKView: MTKView {
    weak var controller: SceneController?

    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        controller?.mouseDown(at: point(of: event), clicks: event.clickCount)
    }

    override func mouseDragged(with event: NSEvent) {
        controller?.mouseDragged(to: point(of: event), dx: event.deltaX, dy: event.deltaY)
    }

    override func mouseUp(with event: NSEvent) {
        controller?.mouseUp()
    }

    /// The event's position in view points with the origin at the top left, like SwiftUI.
    private func point(of event: NSEvent) -> CGPoint {
        let local = convert(event.locationInWindow, from: nil)
        return CGPoint(x: local.x, y: bounds.height - local.y)
    }

    override func rightMouseDragged(with event: NSEvent) {
        controller?.drag(dx: event.deltaX, dy: event.deltaY)
    }

    override func scrollWheel(with event: NSEvent) {
        controller?.scroll(event)
    }

    override func magnify(with event: NSEvent) {
        controller?.magnify(event.magnification)
    }
}

@MainActor
final class SceneController: NSObject, MTKViewDelegate {
    /// SelfTest's key test posts events to a window that may not be key, and those events
    /// do not change the hardware modifier state, so it reads modifiers from the events.
    static var testMode = false
    static let instances = NSHashTable<SceneController>.weakObjects()

    let item: SplatItem
    let model: ViewerModel
    private weak var library: Library?
    weak var view: SplatMTKView?
    private var commandQueue: MTLCommandQueue?
    private(set) var scene: SplatScene?
    var camera = FlyCamera()
    private var keys = KeyState()
    private var keyMonitor: Any?
    private var resignObserver: NSObjectProtocol?
    private var loadTask: Task<Void, Never>?
    private var loadWork: Task<SplatScene, Error>?
    private var active = false
    private var lastFrameTime: CFTimeInterval = 0
    private var lastInteraction: CFTimeInterval = 0
    private var framesRendered = 0
    private var thumbnailRequested = false
    private var eventFlags: NSEvent.ModifierFlags = []
    /// The last place the camera stood in the open, with nothing solid between it and here.
    private var openAnchor: SIMD3<Float>?
    private var clipDistance: Float = 0
    private var appliedEdits = SceneEdits()
    private var applyingEdits = false
    var editDrag: EditDrag?
    var editDragStarted = false

    var debugPose: FlyCamera.Pose { camera.pose }
    var debugHeldKeys: Set<MoveKey> { keys.held }
    var isReady: Bool { scene != nil }

    init(item: SplatItem, model: ViewerModel, library: Library) {
        self.item = item
        self.model = model
        self.library = library
        super.init()
        model.controller = self
        Self.instances.add(self)
    }

    // MARK: Lifecycle

    func attach(_ view: SplatMTKView) {
        self.view = view
        view.controller = self
        guard let device = view.device ?? MTLCreateSystemDefaultDevice() else {
            model.phase = .failed("This Mac has no Metal graphics.")
            return
        }
        view.device = device
        commandQueue = device.makeCommandQueue()
        view.colorPixelFormat = SplatScene.colorFormat
        view.depthStencilPixelFormat = SplatScene.depthFormat
        view.sampleCount = 1
        view.clearColor = SplatScene.clearColor
        view.preferredFramesPerSecond = 60
        view.enableSetNeedsDisplay = false
        view.isPaused = true
        view.delegate = self
        installKeyMonitor()
        load(device: device)
    }

    func detach() {
        loadTask?.cancel()
        loadWork?.cancel()
        if let keyMonitor { NSEvent.removeMonitor(keyMonitor) }
        keyMonitor = nil
        if let resignObserver { NotificationCenter.default.removeObserver(resignObserver) }
        resignObserver = nil
        view?.isPaused = true
        view?.delegate = nil
        scene = nil
    }

    /// Only the tab on screen renders and listens to the keyboard.
    func setActive(_ isActive: Bool) {
        guard isActive != active else { return }
        active = isActive
        keys.releaseAll()
        view?.isPaused = !isActive
        guard isActive else { return }
        lastFrameTime = CACurrentMediaTime()
        touch()
        DispatchQueue.main.async { [weak self] in
            guard let view = self?.view else { return }
            view.window?.makeFirstResponder(view)
        }
    }

    private func load(device: MTLDevice) {
        let url = item.url
        let total = SplatFile.expectedCount(url)
        let model = self.model
        model.phase = .loading(read: 0, total: total)
        let work = Task.detached(priority: .userInitiated) {
            try await SplatScene.load(url: url, device: device) { read in
                Task { @MainActor in
                    if case .loading = model.phase { model.phase = .loading(read: read, total: total) }
                }
            }
        }
        loadWork = work
        loadTask = Task { [weak self] in
            do {
                let scene = try await work.value
                guard let self, !Task.isCancelled else { return }
                self.sceneLoaded(scene)
            } catch is CancellationError {
            } catch {
                model.phase = .failed(error.localizedDescription)
            }
        }
    }

    private func sceneLoaded(_ scene: SplatScene) {
        self.scene = scene
        camera.configure(bounds: scene.bounds, start: scene.start)
        model.splatCount = scene.count
        model.startViewFrame = scene.start?.frame
        model.phase = .ready
        model.editor.attach(room: scene.room, splat: item.url)
        openAnchor = camera.pose.position
        library?.recordLoaded(item.id, splatCount: scene.count)
        touch()
    }

    // MARK: Input

    private func installKeyMonitor() {
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp, .flagsChanged]) { [weak self] event in
            guard let self else { return event }
            return MainActor.assumeIsolated { self.handle(event) ? nil : event }
        }
        resignObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.didResignKeyNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.keys.releaseAll() }
        }
    }

    /// True when the event was used here (so it does not also beep or reach a menu).
    private func handle(_ event: NSEvent) -> Bool {
        guard active, let view, let window = view.window, window.isKeyWindow || Self.testMode,
              event.window === window, !(window.firstResponder is NSText) else { return false }
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        eventFlags = flags
        switch event.type {
        case .flagsChanged:
            if !flags.contains(.command) && !Self.testMode { keys.commandReleased() }
            touch()
            return false
        case .keyDown, .keyUp:
            let isDown = event.type == .keyDown
            if isDown, handleEditKey(event, flags: flags) { return true }
            if let key = MoveKey(keyCode: event.keyCode) {
                // ⌘W, ⌘Q and the like are the app's, not movement.
                if key.isLetter && !flags.isDisjoint(with: [.command, .control, .option]) { return false }
                if isDown { keys.press(key) } else { keys.release(key) }
                touch()
                return true
            }
            guard isDown, flags.isDisjoint(with: [.command, .control, .option]) else { return false }
            switch event.charactersIgnoringModifiers?.lowercased() {
            case "r":
                resetView()
                return true
            case "h", "?", "/":
                withAnimation(.easeOut(duration: 0.2)) { model.showHelp.toggle() }
                return true
            default:
                return false
            }
        default:
            return false
        }
    }

    /// Drag to look around, like grabbing the view and moving it.
    func drag(dx: CGFloat, dy: CGFloat) {
        camera.look(yaw: Float(dx) * 0.0045, pitch: Float(dy) * 0.0045)
        touch()
    }

    /// Two fingers up (or the wheel away from you) walks forward; sideways steps aside.
    func scroll(_ event: NSEvent) {
        let natural = event.isDirectionInvertedFromDevice
        let vertical = Float(natural ? -event.scrollingDeltaY : event.scrollingDeltaY)
        let horizontal = Float(natural ? event.scrollingDeltaX : -event.scrollingDeltaX)
        let scale: Float = event.hasPreciseScrollingDeltas ? 0.006 : 0.25
        camera.step(forward: vertical * scale, right: horizontal * scale)
        touch()
    }

    func magnify(_ amount: CGFloat) {
        camera.step(forward: Float(amount) * 2.5, right: 0)
        touch()
    }

    func resetView() {
        camera.reset()
        openAnchor = camera.pose.position
        clipDistance = 0
        touch()
    }

    func touch() {
        lastInteraction = CACurrentMediaTime()
        if let view, view.preferredFramesPerSecond != 60 { view.preferredFramesPerSecond = 60 }
    }

    // MARK: Drawing

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        let dt = Float(min(max(now - lastFrameTime, 0), 0.05))
        lastFrameTime = now
        guard let commandQueue, let commandBuffer = commandQueue.makeCommandBuffer(),
              let drawable = view.currentDrawable else { return }
        guard let scene else {
            // Not loaded yet: just clear to the background colour.
            if let pass = view.currentRenderPassDescriptor,
               let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) {
                encoder.endEncoding()
                commandBuffer.present(drawable)
            }
            commandBuffer.commit()
            return
        }

        let flags = Self.testMode ? eventFlags : NSEvent.modifierFlags
        camera.advance(dt: dt, input: FlyCamera.Input(held: keys.held, flags: flags))
        if !keys.held.isEmpty || camera.isMoving { lastInteraction = now }
        // Keep sorting results coming in when idle, without spinning at full rate.
        let fps = now - lastInteraction < 1.5 ? 60 : 15
        if view.preferredFramesPerSecond != fps { view.preferredFramesPerSecond = fps }

        updateSeeThrough(dt: dt, grid: scene.grid)
        syncEdits(scene)
        updateGizmo()

        let size = view.drawableSize
        let rendered = (try? scene.render(camera: camera, near: clipDistance > 0 ? clipDistance : nil,
                                          width: Int(size.width), height: Int(size.height),
                                          colorTexture: drawable.texture,
                                          depthTexture: view.depthStencilTexture,
                                          commandBuffer: commandBuffer)) ?? false
        if rendered {
            commandBuffer.present(drawable)
            framesRendered += 1
        }
        commandBuffer.commit()
        if framesRendered >= 30 && !thumbnailRequested {
            thumbnailRequested = true
            saveThumbnail(scene)
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    /// Looking past what the camera has walked into. While the camera moves through the
    /// open, its position is the anchor. Once it steps into a wall or a piece of furniture
    /// (or out the other side), the anchor stays where it last stood in the open, and
    /// everything nearer than the anchor is left out, so the room still shows as if that
    /// wall or object were not there. Turn away and nothing is hidden.
    private func updateSeeThrough(dt: Float, grid: OccupancyGrid?) {
        guard let grid else { return }
        let position = camera.pose.position
        if let anchor = openAnchor {
            if grid.isOpen(position) && grid.isClear(from: anchor, to: position) { openAnchor = position }
        } else {
            openAnchor = position
        }
        let margin = grid.cell * 0.8
        let ahead = simd_dot((openAnchor ?? position) - position, camera.lookDirection(camera.pose))
        let target = ahead > margin ? ahead - margin * 0.5 : 0
        clipDistance += (target - clipDistance) * (1 - exp(-dt * 14))
        if clipDistance < camera.near { clipDistance = 0 }
    }

    /// Hands the latest edits to the scene, one rebuild at a time.
    private func syncEdits(_ scene: SplatScene) {
        guard scene.room != nil, !applyingEdits, model.editor.edits != appliedEdits else { return }
        applyingEdits = true
        let edits = model.editor.edits
        Task { [weak self] in
            await scene.apply(edits)
            guard let self else { return }
            self.appliedEdits = edits
            self.applyingEdits = false
            self.touch()
        }
    }

    /// Test and debug hooks.
    var debugClipDistance: Float { clipDistance }
    var debugAppliedEdits: SceneEdits { appliedEdits }
    var debugCommandQueue: MTLCommandQueue? { commandQueue }

    /// A picture from the starting camera for the library's card.
    private func saveThumbnail(_ scene: SplatScene) {
        guard let commandQueue, let library else { return }
        let id = item.id
        let url = library.thumbnailURL(for: id)
        let camera = self.camera
        let start = camera.startPose
        DispatchQueue.global(qos: .utility).async {
            guard let image = scene.snapshot(camera: camera, pose: start, width: 640, height: 400,
                                             commandQueue: commandQueue),
                  let png = image.pngData() else { return }
            try? png.write(to: url, options: .atomic)
            DispatchQueue.main.async { library.thumbnailSaved(id) }
        }
    }
}
