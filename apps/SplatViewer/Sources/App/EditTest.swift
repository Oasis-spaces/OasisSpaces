import AppKit
import simd

/// SPLATVIEWER_EDITTEST_SPLAT=<splat> SPLATVIEWER_EDITTEST_OUT=<folder>: opens the splat in
/// the real window and edits it the way a person would, through the same calls the mouse
/// and keys make: turns ⌘E on, clicks the bed and deletes it, drags the wardrobe along the
/// floor and by its handles, repaints a wall, drops a sofa, undoes everything, then walks
/// backwards through the wall behind the start. Logs each step and renders PNGs.
@MainActor
enum EditTest {
    static func run(splat: URL, output: URL) {
        SceneController.testMode = true
        try? FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        Library.shared.add([splat])
        Task { @MainActor in
            var log: [String] = []
            @MainActor func finish() {
                try? log.joined(separator: "\n").appending("\n")
                    .write(to: output.appendingPathComponent("edittest.txt"), atomically: true, encoding: .utf8)
                NSApp.terminate(nil)
            }
            @MainActor func controller() -> SceneController? {
                SceneController.instances.allObjects.first { $0.item.path == splat.standardizedFileURL.path && $0.isReady }
            }
            for _ in 0..<300 where controller() == nil { try? await Task.sleep(for: .milliseconds(100)) }
            guard let c = controller(), let view = c.view, let window = view.window,
                  let room = c.scene?.room else {
                log.append("not ready, or no room model beside the splat")
                return finish()
            }
            let editor = c.model.editor
            let size = view.bounds.size
            log.append("view \(Int(size.width))×\(Int(size.height)); objects \(room.objects.map { "\($0.id) \($0.label)" }); walls \(room.walls.map(\.id))")

            @MainActor func post(_ keyCode: UInt16, _ flags: NSEvent.ModifierFlags = [], _ chars: String = "", up: Bool = false) {
                if let event = NSEvent.keyEvent(with: up ? .keyUp : .keyDown, location: .zero, modifierFlags: flags,
                                                timestamp: ProcessInfo.processInfo.systemUptime,
                                                windowNumber: window.windowNumber, context: nil,
                                                characters: chars, charactersIgnoringModifiers: chars,
                                                isARepeat: false, keyCode: keyCode) {
                    NSApp.postEvent(event, atStart: false)
                }
            }
            @MainActor func pause(_ ms: Int) async { try? await Task.sleep(for: .milliseconds(ms)) }
            /// Waits for the scene to catch up with the edits.
            @MainActor func settle() async -> Double {
                let started = Date()
                for _ in 0..<300 where c.debugAppliedEdits != editor.edits { await pause(50) }
                let seconds = Date().timeIntervalSince(started)
                await pause(400)
                return seconds
            }
            @MainActor func shot(_ name: String, clip: Bool = true) async {
                guard let scene = c.scene, let queue = c.debugCommandQueue else { return }
                let camera = c.camera
                let near: Float? = clip && c.debugClipDistance > 0 ? c.debugClipDistance : nil
                let image = await Task.detached {
                    scene.snapshot(camera: camera, near: near, width: 960, height: 600, commandQueue: queue)?.pngData()
                }.value
                try? image?.write(to: output.appendingPathComponent("\(name).png"))
                log.append("  shot \(name): \(image == nil ? "FAILED" : "ok")\(near.map { String(format: " (clip %.2f)", $0) } ?? "")")
            }
            @MainActor func screen(_ scene: SIMD3<Float>) -> CGPoint? {
                // Snapshots are 960×600; the view may differ, so project for the view.
                c.camera.project(room.toSplat(scene), in: size)
            }
            /// Turns the camera (where it stands) to put a scene point in the middle of the view.
            @MainActor func face(_ target: SIMD3<Float>) {
                var best: (FlyCamera.Pose, CGFloat)?
                for yawStep in 0..<120 {
                    for pitchStep in -16...8 {
                        var pose = c.camera.pose
                        pose.yaw = Float(yawStep) * 3 * .pi / 180
                        pose.pitch = Float(pitchStep) * 3 * .pi / 180
                        c.camera.pose = pose
                        guard let p = c.camera.project(room.toSplat(target), in: size) else { continue }
                        let d = hypot(p.x - size.width / 2, p.y - size.height / 2)
                        if best == nil || d < best!.1 { best = (pose, d) }
                    }
                }
                if let best { c.camera.pose = best.0 }
                c.touch()
            }
            @MainActor func captureWindow(_ name: String) async {
                guard let content = window.contentView, let scene = c.scene, let queue = c.debugCommandQueue,
                      let rep = content.bitmapImageRepForCachingDisplay(in: content.bounds) else { return }
                content.cacheDisplay(in: content.bounds, to: rep)
                let scale = window.backingScaleFactor
                let frame = view.convert(view.bounds, to: content)
                let camera = c.camera
                let width = Int(frame.width), height = Int(frame.height)
                let splatImage = await Task.detached {
                    scene.snapshot(camera: camera, width: width, height: height, commandQueue: queue)
                }.value
                // Composited afterwards: the overlay's plain background is where the splat shows.
                try? rep.cgImage?.pngData()?.write(to: output.appendingPathComponent("\(name)-overlay.png"))
                try? splatImage?.pngData()?.write(to: output.appendingPathComponent("\(name)-splat.png"))
                try? "\(frame.minX) \(content.bounds.height - frame.maxY) \(frame.width) \(frame.height) \(scale)"
                    .write(to: output.appendingPathComponent("\(name)-frame.txt"), atomically: true, encoding: .utf8)
                log.append("  window \(name): overlay \(rep.pixelsWide)×\(rep.pixelsHigh), splat \(splatImage == nil ? "FAILED" : "ok")")
            }
            @MainActor func describeSelection() -> String {
                guard editor.selection != nil else { return "nothing selected" }
                let s = editor.selectionSize ?? .zero
                return String(format: "%@ %.2f×%.2f×%.2f m yaw %.0f°", editor.selectionTitle, s.x, s.y, s.z,
                              editor.selectionYaw * 180 / .pi)
            }
            @MainActor func handle(_ wanted: Gizmo.Handle) -> CGPoint? {
                editor.gizmo?.handles.first { $0.handle == wanted }?.point
            }
            @MainActor func middle(_ object: RoomModel.Object) -> SIMD3<Float> {
                SIMD3(object.centre.x, object.centre.y, (object.min.z + object.max.z) / 2)
            }

            for wall in room.walls { log.append("paint check " + (c.scene?.debugPaint(wall.id) ?? "")) }
            if ProcessInfo.processInfo.environment["SPLATVIEWER_EDITTEST_PAINTONLY"] != nil { return finish() }
            await pause(800)
            await shot("e0-start")

            // ⌘E turns editing on.
            post(14, .command, "e")
            await pause(300)
            log.append("⌘E: editing \(editor.isEditing)")

            // Click the bed, delete it.
            if let bed = room.objects.first(where: { $0.label == "bed" }) {
                face(middle(bed))
                await pause(200)
                if let point = screen(middle(bed)) {
                    c.mouseDown(at: point, clicks: 1)
                    c.mouseUp()
                }
                log.append("click bed: \(describeSelection())")
                await pause(300)
                await captureWindow("ui-bed-selected")
                post(51)
                let took = await settle()
                log.append(String(format: "⌫: removed %@, scene caught up in %.2f s", "\(editor.edits.removed)", took))
                await shot("e2-bed-removed")
            }

            // The wardrobe: drag it 50 cm along the floor, widen it by a corner, raise it, turn it.
            if let wardrobe = room.objects.first(where: { $0.label == "wardrobe" }) {
                face(middle(wardrobe))
                await pause(200)
                guard let point = screen(middle(wardrobe)) else { return finish() }
                c.mouseDown(at: point, clicks: 1)
                log.append("click wardrobe: \(describeSelection())")
                if let hit = c.floorPoint(at: point) {
                    let box = editor.box()!
                    // Along the room, away from the nearest wall end.
                    let direction: SIMD2<Float> = box.centre.x > room.centre.x ? SIMD2(-1, 0) : SIMD2(1, 0)
                    for step in 1...10 {
                        let target = hit + direction * 0.5 * room.metre * Float(step) / 10
                        if let p = screen(SIMD3(target.x, target.y, room.floorZ)) {
                            c.mouseDragged(to: p, dx: 0, dy: 0)
                            await pause(30)
                        }
                    }
                    c.mouseUp()
                    let moved = (editor.box()!.centre - box.centre) / room.metre
                    let took = await settle()
                    log.append(String(format: "drag: moved (%.2f, %.2f) m; %@; caught up in %.2f s",
                                      moved.x, moved.y, describeSelection(), took))
                    await shot("e3-wardrobe-moved")
                }
                face(middle(room.object(wardrobe.id).map { _ in
                    let b = editor.box()!
                    return RoomModel.Object(id: "", label: "", min: SIMD3(b.centre.x, b.centre.y, b.bottom),
                                            max: SIMD3(b.centre.x, b.centre.y, b.top))
                }!))
                await pause(250)
                if let corner = handle(.corner(1, 1)), let box = editor.box() {
                    c.mouseDown(at: corner, clicks: 1)
                    let far = box.centre + box.axisX * (box.half.x + 0.2 * room.metre) + box.axisY * box.half.y
                    if let p = screen(SIMD3(far.x, far.y, room.floorZ)) { c.mouseDragged(to: p, dx: 0, dy: 0) }
                    c.mouseUp()
                    log.append("corner handle: \(describeSelection()) (wider by 40 cm, depth kept)")
                } else {
                    log.append("corner handle not on screen; handles \(editor.gizmo?.handles.map(\.point) ?? [])")
                }
                if let top = handle(.height) {
                    c.mouseDown(at: top, clicks: 1)
                    c.mouseDragged(to: CGPoint(x: top.x, y: top.y - 30), dx: 0, dy: -30)
                    c.mouseUp()
                    log.append("height handle dragged up 30 pt: \(describeSelection())")
                }
                if let rotate = handle(.rotate), let box = editor.box(), let start = c.floorPoint(at: rotate) {
                    c.mouseDown(at: rotate, clicks: 1)
                    let arm = start - box.centre
                    let angle: Float = 30 * .pi / 180
                    let turned = box.centre + SIMD2(arm.x * cos(angle) - arm.y * sin(angle), arm.x * sin(angle) + arm.y * cos(angle))
                    if let p = screen(SIMD3(turned.x, turned.y, room.floorZ)) { c.mouseDragged(to: p, dx: 0, dy: 0) }
                    c.mouseUp()
                    log.append("rotate handle turned 30°: \(describeSelection())")
                }
                post(30, [], "]")
                await pause(100)
                log.append("]: \(describeSelection())")
                let took = await settle()
                log.append(String(format: "caught up in %.2f s; labels %@", took, "\(editor.gizmo?.labels.map(\.text) ?? [])"))
                await shot("e4-wardrobe-resized-turned")
            }

            // Click a wall and paint it sage.
            if let wall = room.walls.first(where: { wall in
                // One the camera can face from where it stands, well inside its length.
                simd_dot(SIMD3(room.centre.x, room.centre.y, wall.centre.z) - wall.centre, wall.inward) > 0
            }) {
                let target = SIMD3(wall.centre.x, wall.centre.y, room.floorZ + room.height * 0.6) + wall.inward * 0.02 * room.metre
                face(target)
                await pause(200)
                if let p = screen(target) {
                    c.mouseDown(at: p, clicks: 1)
                    c.mouseUp()
                }
                log.append("click wall \(wall.id): \(describeSelection()) selection \(String(describing: editor.selection))")
                await shot("e5-wall-before")
                editor.setColour("#B4C2A5")
                var took = await settle()
                log.append(String(format: "paint: %@ caught up in %.2f s", "\(editor.edits.wallColours)", took))
                await shot("e5-wall-painted")
                editor.paintAllWalls("#B4C2A5")
                took = await settle()
                let facing = c.camera.pose
                c.camera.reset()
                c.touch()
                await pause(300)
                log.append(String(format: "paint all walls: %d painted, caught up in %.2f s", editor.edits.wallColours.count, took))
                await shot("e5-all-walls-start-view")
                c.camera.pose = facing
            }

            // Drop a sofa from the library onto the middle of the floor.
            editor.showLibrary = true
            face(SIMD3(room.centre.x, room.centre.y, room.floorZ))
            await pause(200)
            c.dropFurniture(.sofa, at: CGPoint(x: size.width / 2, y: size.height / 2))
            let took = await settle()
            log.append(String(format: "drop sofa: %@; caught up in %.2f s", describeSelection(), took))
            if let item = editor.edits.added.first {
                log.append("  sofa at (\(item.centre.x), \(item.centre.y)) scene units")
            }
            editor.setColour("#3F6E73")
            _ = await settle()
            await shot("e6-sofa-added")

            // The window with its panels, the splat drawn in underneath (a window capture
            // does not include Metal content).
            await pause(500)
            await captureWindow("ui-editing")

            // ⌘Z until nothing is left to undo.
            editor.showLibrary = false
            var undos = 0
            while editor.canUndo && undos < 200 {
                post(6, .command, "z")
                await pause(40)
                undos += 1
            }
            let undone = await settle()
            log.append(String(format: "⌘Z ×%d: edits empty %@, caught up in %.2f s", undos, "\(editor.edits.isEmpty)", undone))
            await pause(900)
            log.append("edits file after undo exists: \(FileManager.default.fileExists(atPath: SceneEdits.file(for: splat).path))")
            post(14, .command, "e")
            await pause(200)
            log.append("⌘E: editing \(editor.isEditing)")

            // Walk backwards from the start through whatever is behind.
            c.resetView()
            await pause(300)
            let startScene = room.toScene(c.camera.pose.position)
            log.append(String(format: "see-through: start at scene (%.2f, %.2f, %.2f)", startScene.x, startScene.y, startScene.z))
            await shot("s0-start")
            post(125)
            var passed = false
            for tick in 0..<40 {
                await pause(250)
                let position = c.camera.pose.position
                let solid = c.scene?.grid?.isSolid(position) ?? false
                let scenePosition = room.toScene(position)
                let outside = abs(scenePosition.x - room.centre.x) > room.half.x || abs(scenePosition.y - room.centre.y) > room.half.y
                if tick % 2 == 1 || c.debugClipDistance > 0 {
                    log.append(String(format: "  %.2f s: scene (%.2f, %.2f) solid %@ outside %@ clip %.2f m",
                                      Float(tick + 1) / 4, scenePosition.x, scenePosition.y, "\(solid)", "\(outside)",
                                      c.debugClipDistance / room.metre))
                }
                if outside && c.debugClipDistance > 0 {
                    // Half a metre beyond the wall.
                    if passed { break }
                    passed = true
                }
            }
            post(125, up: true)
            await pause(900)
            log.append(String(format: "stopped: clip %.2f m", c.debugClipDistance / room.metre))
            await shot("s1-backed-through-with-clip")
            await shot("s1-backed-through-without-clip", clip: false)
            // Turn around: nothing should be hidden looking away from the room.
            var pose = c.camera.pose
            pose.yaw += .pi
            c.camera.pose = pose
            c.touch()
            await pause(900)
            log.append(String(format: "turned round: clip %.2f m", c.debugClipDistance / room.metre))
            // Walk back in.
            c.resetView()
            await pause(900)
            log.append(String(format: "reset: clip %.2f m", c.debugClipDistance / room.metre))
            finish()
        }
    }
}
