import AppKit
import simd

/// What a mouse drag in the view is doing.
enum EditDrag {
    case look
    case move(grab: SIMD2<Float>, start: SIMD2<Float>)
    case rotate(startAngle: Float, startYaw: Float)
    case resize(signs: SIMD2<Float>)
    case height(startY: CGFloat, startHeight: Float)
}

/// Editing in the view: picking things, dragging, turning and resizing them with the
/// selection's handles, and the geometry of the overlay that shows them.
extension SceneController {
    private var editor: EditorModel { model.editor }
    private var room: RoomModel? { scene?.room }
    private var viewSize: CGSize { view?.bounds.size ?? .zero }

    // MARK: Mouse

    func mouseDown(at point: CGPoint, clicks: Int) {
        editDragStarted = false
        guard editor.isEditing, room != nil else {
            editDrag = .look
            return
        }
        if let handle = handle(at: point), let box = editor.box() {
            switch handle {
            case .corner(let sx, let sy):
                editDrag = .resize(signs: SIMD2(sx, sy))
            case .height:
                editDrag = .height(startY: point.y, startHeight: box.top - box.bottom)
            case .rotate:
                let hit = floorPoint(at: point) ?? box.centre
                editDrag = .rotate(startAngle: atan2(hit.y - box.centre.y, hit.x - box.centre.x),
                                   startYaw: editor.selectionYaw)
            }
            return
        }
        let picked = pick(at: point)
        editor.selection = picked
        switch picked {
        case .object, .added:
            let centre = editor.box()?.centre ?? .zero
            let hit = floorPoint(at: point) ?? centre
            editDrag = .move(grab: centre - hit, start: centre)
        default:
            editDrag = .look
        }
        touch()
    }

    func mouseDragged(to point: CGPoint, dx: CGFloat, dy: CGFloat) {
        guard let drag = editDrag else {
            self.drag(dx: dx, dy: dy)
            return
        }
        if case .look = drag {
            self.drag(dx: dx, dy: dy)
            return
        }
        guard let room else { return }
        if !editDragStarted {
            editDragStarted = true
            editor.beginChange()
        }
        switch drag {
        case .look:
            break
        case .move(let grab, let start):
            guard let hit = floorPoint(at: point), let box = editor.box() else { return }
            editor.setCentre(inside(room, hit + grab, half: box.reach, allowing: start))
        case .rotate(let startAngle, let startYaw):
            guard let box = editor.box(), let hit = floorPoint(at: point) else { return }
            var yaw = startYaw + atan2(hit.y - box.centre.y, hit.x - box.centre.x) - startAngle
            if NSEvent.modifierFlags.contains(.shift) {
                yaw = (yaw / (.pi / 12)).rounded() * (.pi / 12)   // 15° steps
            }
            editor.setYaw(yaw, continuous: true)
        case .resize(let signs):
            guard let box = editor.box(), let size = editor.selectionSize, let hit = floorPoint(at: point) else { return }
            let d = hit - box.centre
            var half = SIMD2(abs(simd_dot(d, box.axisX)), abs(simd_dot(d, box.axisY)))
            if NSEvent.modifierFlags.contains(.shift) {
                // Keep the proportions of the footprint.
                let k = max(half.x / max(box.half.x, 1e-6), half.y / max(box.half.y, 1e-6))
                half = box.half * k
            }
            let width = signs.x != 0 ? half.x * 2 / room.metre : size.x
            let depth = signs.y != 0 ? half.y * 2 / room.metre : size.y
            editor.setSize(SIMD3(max(width, 0.1), max(depth, 0.1), size.z), continuous: true)
        case .height(let startY, let startHeight):
            guard let size = editor.selectionSize, let box = editor.box() else { return }
            // Screen pixels to scene height at the object's distance.
            let base = room.toSplat(SIMD3(box.centre.x, box.centre.y, box.bottom))
            let top = room.toSplat(SIMD3(box.centre.x, box.centre.y, box.bottom + startHeight))
            guard let a = camera.project(base, in: viewSize), let b = camera.project(top, in: viewSize) else { return }
            let pixelsPerUnit = max(abs(Float(a.y - b.y)) / startHeight, 1e-4)
            let height = startHeight + Float(startY - point.y) / pixelsPerUnit
            editor.setSize(SIMD3(size.x, size.y, max(height / room.metre, 0.05)), continuous: true)
        }
        touch()
    }

    func mouseUp() {
        editDrag = nil
        editDragStarted = false
    }

    // MARK: Keys

    /// Keys that only mean something while editing (and ⌘E, which starts and ends it).
    func handleEditKey(_ event: NSEvent, flags: NSEvent.ModifierFlags) -> Bool {
        let key = event.charactersIgnoringModifiers?.lowercased()
        if flags.intersection([.command, .control, .option]) == .command, key == "e", editor.hasRoom {
            editor.isEditing.toggle()
            touch()
            return true
        }
        guard editor.isEditing else { return false }
        if flags.contains(.command), key == "z" {
            flags.contains(.shift) ? editor.redo() : editor.undo()
            touch()
            return true
        }
        guard flags.isDisjoint(with: [.command, .control, .option]) else { return false }
        switch event.keyCode {
        case 51, 117:   // delete, forward delete
            editor.removeSelection()
        case 53:        // escape
            if editor.showLibrary { editor.showLibrary = false } else { editor.selection = nil }
        default:
            switch key {
            case "[": editor.rotateSelection(by: -.pi / 12)
            case "]": editor.rotateSelection(by: .pi / 12)
            default: return false
            }
        }
        touch()
        return true
    }

    // MARK: Adding

    /// Drops a piece of furniture where the pointer meets the floor.
    func dropFurniture(_ kind: FurnitureKind, at point: CGPoint) {
        guard let room else { return }
        let centre = floorPoint(at: point) ?? pointInFront(room)
        editor.isEditing = true
        editor.add(kind, at: inside(room, centre, half: SIMD2(kind.size.x, kind.size.y) * room.metre / 2))
        touch()
    }

    /// Places a piece of furniture on the floor a step and a half ahead.
    func addFurnitureInFront(_ kind: FurnitureKind) {
        guard let room else { return }
        editor.add(kind, at: inside(room, pointInFront(room), half: SIMD2(kind.size.x, kind.size.y) * room.metre / 2))
        touch()
    }

    /// A footprint centre moved so the footprint (half-extents along x and y) stays in the
    /// room, without pulling back something that already stood past the edge.
    private func inside(_ room: RoomModel, _ centre: SIMD2<Float>, half: SIMD2<Float>,
                        allowing start: SIMD2<Float>? = nil) -> SIMD2<Float> {
        var lo = room.centre - room.half + simd_min(half, room.half)
        var hi = room.centre + room.half - simd_min(half, room.half)
        if let start {
            lo = simd_min(lo, start)
            hi = simd_max(hi, start)
        }
        return simd_clamp(centre, lo, hi)
    }

    private func pointInFront(_ room: RoomModel) -> SIMD2<Float> {
        let ahead = camera.pose.position + camera.heading(camera.pose) * 1.6 * room.metre
        let scene = room.toScene(ahead)
        return SIMD2(scene.x, scene.y)
    }

    // MARK: Picking

    /// Where the ray through a view point meets the floor, in scene units.
    func floorPoint(at point: CGPoint) -> SIMD2<Float>? {
        guard let room else { return nil }
        let ray = camera.ray(through: point, in: viewSize)
        let origin = room.toScene(ray.origin), direction = room.toScene(ray.direction)
        guard direction.z < -1e-4 else { return nil }
        let t = (room.floorZ - origin.z) / direction.z
        guard t > 0 else { return nil }
        let hit = origin + direction * t
        return SIMD2(hit.x, hit.y)
    }

    /// The nearest object, added piece or wall under a view point.
    private func pick(at point: CGPoint) -> EditorModel.Selection? {
        guard let room else { return nil }
        let ray = camera.ray(through: point, in: viewSize)
        let origin = room.toScene(ray.origin), direction = room.toScene(ray.direction)
        var best: (EditorModel.Selection, Float)?
        func consider(_ selection: EditorModel.Selection, _ t: Float?) {
            guard let t, best == nil || t < best!.1 else { return }
            best = (selection, t)
        }
        for object in room.objects where !editor.edits.removed.contains(object.id) {
            consider(.object(object.id), object.box(editor.edits.placement(object.id)).hit(origin: origin, direction: direction))
        }
        for item in editor.edits.added {
            consider(.added(item.id), item.box(metre: room.metre, floorZ: room.floorZ).hit(origin: origin, direction: direction))
        }
        for wall in room.walls {
            let denominator = simd_dot(direction, wall.inward)
            guard abs(denominator) > 1e-5 else { continue }
            let t = simd_dot(wall.centre - origin, wall.inward) / denominator
            guard t > 0 else { continue }
            let hit = origin + direction * t
            if abs(simd_dot(hit - wall.centre, wall.along)) <= wall.halfLength,
               hit.z >= room.floorZ, hit.z <= room.floorZ + room.height {
                consider(.wall(wall.id), t)
            }
        }
        return best?.0
    }

    private func handle(at point: CGPoint) -> Gizmo.Handle? {
        guard let gizmo = editor.gizmo else { return nil }
        return gizmo.handles.first { hypot($0.point.x - point.x, $0.point.y - point.y) < 14 }?.handle
    }

    // MARK: Overlay

    /// Recomputes the selection overlay in view points.
    func updateGizmo() {
        guard editor.isEditing, let room, let selection = editor.selection else {
            if editor.gizmo != nil { editor.gizmo = nil }
            return
        }
        let size = viewSize
        func screen(_ scene: SIMD3<Float>) -> CGPoint? { camera.project(room.toSplat(scene), in: size) }
        var gizmo = Gizmo()
        let m = room.metre
        if let box = editor.box(for: selection) {
            let corners = box.corners.map(screen)
            for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)] {
                if let p = corners[a], let q = corners[b] { gizmo.edges.append([p, q]) }
            }
            let signs: [(Float, Float)] = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            for (index, sign) in signs.enumerated() {
                if let p = corners[index] { gizmo.handles.append((.corner(sign.0, sign.1), p)) }
            }
            if let top = screen(SIMD3(box.centre.x, box.centre.y, box.top)) {
                gizmo.handles.append((.height, top))
            }
            let front = box.centre - box.axisY * (box.half.y + 0.35 * m)
            if let rotate = screen(SIMD3(front.x, front.y, box.bottom)),
               let edge = screen(SIMD3(box.centre.x - box.axisY.x * box.half.y, box.centre.y - box.axisY.y * box.half.y, box.bottom)) {
                gizmo.edges.append([edge, rotate])
                gizmo.handles.append((.rotate, rotate))
            }
            let width = box.half.x * 2 / m, depth = box.half.y * 2 / m, height = (box.top - box.bottom) / m
            func mid(_ a: Int, _ b: Int) -> CGPoint? {
                guard let p = corners[a], let q = corners[b] else { return nil }
                return CGPoint(x: (p.x + q.x) / 2, y: (p.y + q.y) / 2)
            }
            if let p = mid(0, 1) { gizmo.labels.append((Self.length(width), p)) }
            if let p = mid(1, 2) { gizmo.labels.append((Self.length(depth), p)) }
            if let p = mid(1, 5) { gizmo.labels.append((Self.length(height), p)) }
        } else if case .wall(let id) = selection, let wall = room.wall(id) {
            let top = room.floorZ + room.height
            let ends = [-wall.halfLength, wall.halfLength]
            let points = [(ends[0], room.floorZ), (ends[1], room.floorZ), (ends[1], top), (ends[0], top)]
                .map { screen(wall.centre + wall.along * $0.0 + SIMD3(0, 0, $0.1 - wall.centre.z) + wall.inward * 0.01 * m) }
            for index in 0..<4 {
                if let p = points[index], let q = points[(index + 1) % 4] { gizmo.edges.append([p, q]) }
            }
            if let centre = screen(SIMD3(wall.centre.x, wall.centre.y, room.floorZ + room.height / 2)) {
                gizmo.labels.append(("\(Self.length(wall.halfLength * 2 / m)) × \(Self.length(room.height / m))", centre))
            }
        }
        if editor.gizmo != gizmo { editor.gizmo = gizmo }
    }

    static func length(_ metres: Float) -> String {
        metres < 1 ? "\(Int((metres * 100).rounded())) cm" : String(format: "%.2f m", metres)
    }
}
