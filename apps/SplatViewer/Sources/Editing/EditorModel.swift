import Foundation
import Observation
import simd

/// Every change made to a splat in the viewer, saved beside it as <name>.edits.json.
struct SceneEdits: Codable, Equatable {
    var removed: Set<String> = []
    var placements: [String: Placement] = [:]
    var wallColours: [String: String] = [:]
    var added: [AddedItem] = []

    func placement(_ id: String) -> Placement { placements[id] ?? Placement() }
    var isEmpty: Bool { removed.isEmpty && placements.isEmpty && wallColours.isEmpty && added.isEmpty }

    static func file(for splat: URL) -> URL {
        splat.deletingPathExtension().appendingPathExtension("edits.json")
    }

    static func load(for splat: URL) -> SceneEdits? {
        guard let data = try? Data(contentsOf: file(for: splat)) else { return nil }
        return try? JSONDecoder().decode(SceneEdits.self, from: data)
    }
}

/// What the selection overlay draws, in view points (origin top-left).
struct Gizmo: Equatable {
    enum Handle: Equatable {
        case corner(Float, Float)   // footprint corner, by sign along the box's width and depth
        case height
        case rotate
    }
    var edges: [[CGPoint]] = []
    var handles: [(handle: Handle, point: CGPoint)] = []
    var labels: [(text: String, point: CGPoint)] = []

    static func == (a: Gizmo, b: Gizmo) -> Bool {
        a.edges == b.edges && a.handles.map(\.point) == b.handles.map(\.point)
            && a.labels.map(\.text) == b.labels.map(\.text) && a.labels.map(\.point) == b.labels.map(\.point)
    }
}

/// Editing state for one open splat: the edits with undo, the selection and the overlay.
@MainActor
@Observable
final class EditorModel {
    enum Selection: Hashable {
        case object(String)
        case wall(String)
        case added(UUID)
    }

    var isEditing = false {
        didSet {
            if !isEditing {
                selection = nil
                showLibrary = false
            }
        }
    }
    var showLibrary = false
    var selection: Selection?
    private(set) var edits = SceneEdits()
    private(set) var canUndo = false
    private(set) var canRedo = false
    private(set) var hasRoom = false
    var gizmo: Gizmo?

    @ObservationIgnored private(set) var room: RoomModel?
    @ObservationIgnored private var splatURL: URL?
    @ObservationIgnored private var undoStack: [SceneEdits] = []
    @ObservationIgnored private var redoStack: [SceneEdits] = []
    @ObservationIgnored private var saveTask: Task<Void, Never>?

    func attach(room: RoomModel?, splat: URL) {
        self.room = room
        splatURL = splat
        hasRoom = room != nil
        if room != nil, let saved = SceneEdits.load(for: splat) { edits = saved }
    }

    // MARK: Changing

    /// Starts an undoable change; a drag calls this once and `update` many times.
    func beginChange() {
        undoStack.append(edits)
        if undoStack.count > 100 { undoStack.removeFirst() }
        redoStack.removeAll()
        refreshUndo()
    }

    func update(_ change: (inout SceneEdits) -> Void) {
        var next = edits
        change(&next)
        guard next != edits else { return }
        edits = next
        scheduleSave()
    }

    func commit(_ change: (inout SceneEdits) -> Void) {
        beginChange()
        update(change)
    }

    func undo() {
        guard let previous = undoStack.popLast() else { return }
        redoStack.append(edits)
        edits = previous
        dropStaleSelection()
        refreshUndo()
        scheduleSave()
    }

    func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(edits)
        edits = next
        dropStaleSelection()
        refreshUndo()
        scheduleSave()
    }

    private func refreshUndo() {
        canUndo = !undoStack.isEmpty
        canRedo = !redoStack.isEmpty
    }

    private func dropStaleSelection() {
        switch selection {
        case .object(let id) where edits.removed.contains(id): selection = nil
        case .added(let id) where !edits.added.contains(where: { $0.id == id }): selection = nil
        default: break
        }
    }

    private func scheduleSave() {
        saveTask?.cancel()
        guard let splatURL else { return }
        let edits = self.edits
        saveTask = Task {
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled else { return }
            let file = SceneEdits.file(for: splatURL)
            if edits.isEmpty {
                try? FileManager.default.removeItem(at: file)
            } else if let data = try? JSONEncoder().encode(edits) {
                try? data.write(to: file, options: .atomic)
            }
        }
    }

    // MARK: The selection

    var selectionTitle: String {
        guard let room else { return "" }
        switch selection {
        case .object(let id): return (room.object(id)?.label ?? "Object").capitalized
        case .wall: return "Wall"
        case .added(let id): return edits.added.first { $0.id == id }?.kind.label ?? "Furniture"
        case nil: return ""
        }
    }

    /// The selected object's box as it now stands.
    func box(for selection: Selection? = nil) -> FloorBox? {
        guard let room else { return nil }
        switch selection ?? self.selection {
        case .object(let id): return room.object(id)?.box(edits.placement(id))
        case .added(let id): return edits.added.first { $0.id == id }?.box(metre: room.metre, floorZ: room.floorZ)
        default: return nil
        }
    }

    /// Width, depth and height in metres.
    var selectionSize: SIMD3<Float>? {
        guard let room, let box = box() else {
            if case .wall(let id) = selection, let room, let wall = room.wall(id) {
                return SIMD3(wall.halfLength * 2, 0, room.height) / room.metre
            }
            return nil
        }
        return SIMD3(box.half.x * 2, box.half.y * 2, box.top - box.bottom) / room.metre
    }

    var selectionYaw: Float {
        switch selection {
        case .object(let id): edits.placement(id).yaw
        case .added(let id): edits.added.first { $0.id == id }?.yaw ?? 0
        default: 0
        }
    }

    var removedCount: Int { edits.removed.count }

    /// Sets the selection's size in metres (an object is stretched from its filmed size).
    func setSize(_ metres: SIMD3<Float>, continuous: Bool = false) {
        guard let room else { return }
        let size = simd_clamp(metres, SIMD3(repeating: 0.05), SIMD3(repeating: 12))
        let apply: (inout SceneEdits) -> Void = { edits in
            switch self.selection {
            case .object(let id):
                guard let object = room.object(id) else { return }
                var placement = edits.placement(id)
                placement.scale = size * room.metre / object.size
                edits.placements[id] = placement
            case .added(let id):
                guard let index = edits.added.firstIndex(where: { $0.id == id }) else { return }
                edits.added[index].size = size
            default:
                break
            }
        }
        continuous ? update(apply) : commit(apply)
    }

    func setYaw(_ yaw: Float, continuous: Bool = false) {
        let wrapped = atan2(sin(yaw), cos(yaw))
        let apply: (inout SceneEdits) -> Void = { edits in
            switch self.selection {
            case .object(let id):
                var placement = edits.placement(id)
                placement.yaw = wrapped
                edits.placements[id] = placement
            case .added(let id):
                guard let index = edits.added.firstIndex(where: { $0.id == id }) else { return }
                edits.added[index].yaw = wrapped
            default:
                break
            }
        }
        continuous ? update(apply) : commit(apply)
    }

    /// Moves the selection so its footprint's middle is at `centre` (scene units).
    func setCentre(_ centre: SIMD2<Float>) {
        guard let room else { return }
        update { edits in
            switch self.selection {
            case .object(let id):
                guard let object = room.object(id) else { return }
                var placement = edits.placement(id)
                placement.offset = centre - object.centre
                edits.placements[id] = placement
            case .added(let id):
                guard let index = edits.added.firstIndex(where: { $0.id == id }) else { return }
                edits.added[index].centre = centre
            default:
                break
            }
        }
    }

    func rotateSelection(by angle: Float) {
        setYaw(selectionYaw + angle)
    }

    func removeSelection() {
        switch selection {
        case .object(let id): commit { $0.removed.insert(id) }
        case .added(let id): commit { $0.added.removeAll { $0.id == id } }
        default: return
        }
        selection = nil
    }

    func restoreRemoved() {
        commit { $0.removed.removeAll() }
    }

    /// Undoes every change to the selected thing.
    func resetSelection() {
        switch selection {
        case .object(let id): commit { $0.placements[id] = nil }
        case .wall(let id): commit { $0.wallColours[id] = nil }
        case .added(let id):
            commit { edits in
                guard let index = edits.added.firstIndex(where: { $0.id == id }) else { return }
                edits.added[index].size = edits.added[index].kind.size
                edits.added[index].yaw = 0
                edits.added[index].colour = nil
            }
        default: break
        }
    }

    var selectionColour: String? {
        switch selection {
        case .wall(let id): edits.wallColours[id]
        case .added(let id): edits.added.first { $0.id == id }?.colour
        default: nil
        }
    }

    func setColour(_ hex: String?, continuous: Bool = false) {
        let apply: (inout SceneEdits) -> Void = { edits in
            switch self.selection {
            case .wall(let id): edits.wallColours[id] = hex
            case .added(let id):
                guard let index = edits.added.firstIndex(where: { $0.id == id }) else { return }
                edits.added[index].colour = hex
            default: break
            }
        }
        continuous ? update(apply) : commit(apply)
    }

    /// Paints every detected wall one colour (nil puts back the filmed colours).
    func paintAllWalls(_ hex: String?) {
        guard let room else { return }
        commit { edits in
            for wall in room.walls { edits.wallColours[wall.id] = hex }
        }
    }

    @discardableResult
    func add(_ kind: FurnitureKind, at centre: SIMD2<Float>) -> UUID {
        let item = AddedItem(kind: kind, centre: centre, size: kind.size)
        commit { $0.added.append(item) }
        selection = .added(item.id)
        return item.id
    }
}
