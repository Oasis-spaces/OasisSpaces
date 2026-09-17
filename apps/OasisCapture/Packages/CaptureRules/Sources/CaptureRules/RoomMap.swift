import Foundation
import simd

/// A flat surface the phone's tracking found (ARKit or ARCore plane), in
/// world space, y up. Walls and floors come from here; furniture from the
/// segmentation fused with tracked points.
public struct PlaneInfo: Identifiable, Sendable, Equatable {
    public enum Kind: String, Sendable, Codable { case wall, floor, ceiling, table, seat, door, window, unknown }

    public var id: UUID
    public var kind: Kind
    public var vertical: Bool
    public var center: SIMD3<Float>
    /// In-plane axes and the size along each, metres.
    public var xAxis: SIMD3<Float>
    public var zAxis: SIMD3<Float>
    public var extent: SIMD2<Float>

    public init(id: UUID, kind: Kind, vertical: Bool, center: SIMD3<Float>, xAxis: SIMD3<Float>,
                zAxis: SIMD3<Float>, extent: SIMD2<Float>) {
        self.id = id
        self.kind = kind
        self.vertical = vertical
        self.center = center
        self.xAxis = xAxis
        self.zAxis = zAxis
        self.extent = extent
    }

    /// The plane's four corners, world space.
    public var corners: [SIMD3<Float>] {
        let hx = xAxis * (extent.x / 2), hz = zAxis * (extent.y / 2)
        return [center - hx - hz, center + hx - hz, center + hx + hz, center - hx + hz]
    }
}

/// A wall seen from above: a segment on the floor plan.
public struct WallSegment: Identifiable, Sendable, Equatable {
    public var id: UUID
    public var from: SIMD2<Float>
    public var to: SIMD2<Float>
    public var height: Float
    public var kind: PlaneInfo.Kind
}

/// A piece of furniture (or fixture) the scan has placed: an axis-aligned box.
public struct ObjectBox: Identifiable, Sendable, Equatable {
    public var id: String
    public var classId: Int
    public var label: String
    public var group: String
    public var min: SIMD3<Float>
    public var max: SIMD3<Float>
    public var points: Int

    public var center: SIMD3<Float> { (min + max) / 2 }
    public var size: SIMD3<Float> { max - min }
    /// Top-down footprint (x, z).
    public var footprintMin: SIMD2<Float> { SIMD2(min.x, min.z) }
    public var footprintMax: SIMD2<Float> { SIMD2(max.x, max.z) }
}

/// What has been detected in the room so far, for the map and the overlay.
public struct RoomMap: Sendable, Equatable {
    public var planes: [PlaneInfo] = []
    public var objects: [ObjectBox] = []

    public init() {}

    public var walls: [WallSegment] {
        planes.filter { $0.vertical && $0.kind != .door && $0.kind != .window }.map { plane in
            // The more level in-plane axis runs along the wall; the other is its height.
            let xAlong = abs(plane.xAxis.y) <= abs(plane.zAxis.y)
            let along = xAlong ? plane.xAxis : plane.zAxis
            let length = xAlong ? plane.extent.x : plane.extent.y
            let height = xAlong ? plane.extent.y : plane.extent.x
            let c = SIMD2(plane.center.x, plane.center.z)
            let d = simd_normalize(SIMD2(along.x, along.z)) * (length / 2)
            return WallSegment(id: plane.id, from: c - d, to: c + d, height: height, kind: plane.kind)
        }
    }

    public var floors: [PlaneInfo] { planes.filter { !$0.vertical && $0.kind == .floor } }

    /// Extent of everything on the floor plan: (min, max) in x, z.
    public var bounds: (min: SIMD2<Float>, max: SIMD2<Float>)? {
        var lo = SIMD2<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        var any = false
        for plane in planes where plane.kind != .ceiling {
            for c in plane.corners {
                lo = simd_min(lo, SIMD2(c.x, c.z)); hi = simd_max(hi, SIMD2(c.x, c.z)); any = true
            }
        }
        for o in objects {
            lo = simd_min(lo, o.footprintMin); hi = simd_max(hi, o.footprintMax); any = true
        }
        return any ? (lo, hi) : nil
    }
}

/// Builds the room map: keeps the tracking's planes, and turns tracked points
/// labelled by the segmentation into furniture boxes. Points are binned into
/// voxels so a wardrobe seen for a minute does not weigh more than one seen
/// for a second; a box needs enough voxels to be more than a stray label.
public final class RoomMapBuilder {
    public let spec: DetectionSpec
    /// Voxel size for labelled points, metres.
    public var voxel: Float = 0.08
    /// Cell size for grouping voxels into one object, metres.
    public var cell: Float = 0.25
    /// Voxels an object needs before it is placed.
    public var minVoxels = 12
    /// Classes never boxed: walls, floor and ceiling come from planes.
    public var excludedGroups: Set<String> = ["structure", "person"]

    private var planes: [UUID: PlaneInfo] = [:]
    /// voxel key -> class id -> hits
    private var votes: [SIMD3<Int32>: [Int: Int]] = [:]
    private let lock = NSLock()

    public init(spec: DetectionSpec) {
        self.spec = spec
    }

    public func reset() {
        lock.withLock {
            planes = [:]
            votes = [:]
        }
    }

    public func update(plane: PlaneInfo) {
        lock.withLock { planes[plane.id] = plane }
    }

    public func remove(plane id: UUID) {
        lock.withLock { planes[id] = nil }
    }

    /// A tracked point the segmentation put in class `classId` this frame.
    public func add(point: SIMD3<Float>, classId: Int) {
        guard let info = spec.info(classId), info.outline, !excludedGroups.contains(info.group) else { return }
        let key = SIMD3<Int32>(Int32((point.x / voxel).rounded(.down)), Int32((point.y / voxel).rounded(.down)),
                               Int32((point.z / voxel).rounded(.down)))
        lock.withLock { votes[key, default: [:]][classId, default: 0] += 1 }
    }

    public func add(points: [(SIMD3<Float>, Int)]) {
        for (p, c) in points { add(point: p, classId: c) }
    }

    /// The map as it stands: planes, and one box per connected group of
    /// voxels that agree on a class.
    public func build() -> RoomMap {
        let (planes, votes) = lock.withLock { (Array(self.planes.values), self.votes) }
        var map = RoomMap()
        map.planes = planes.sorted { $0.id.uuidString < $1.id.uuidString }

        // Each voxel takes its majority class.
        var byClass: [Int: [SIMD3<Int32>]] = [:]
        for (key, counts) in votes {
            guard let best = counts.max(by: { $0.value < $1.value }), best.value >= 2 else { continue }
            byClass[best.key, default: []].append(key)
        }
        let scale = voxel / cell
        for (classId, voxels) in byClass {
            guard voxels.count >= minVoxels, let info = spec.info(classId) else { continue }
            // Group on a coarse top-down grid, 8-connected, ignoring height so a
            // tall wardrobe seen in pieces stays one object.
            var cells: [SIMD2<Int32>: [SIMD3<Int32>]] = [:]
            for v in voxels {
                let c = SIMD2<Int32>(Int32((Float(v.x) * scale).rounded(.down)), Int32((Float(v.z) * scale).rounded(.down)))
                cells[c, default: []].append(v)
            }
            var seen = Set<SIMD2<Int32>>()
            for start in cells.keys.sorted(by: { ($0.x, $0.y) < ($1.x, $1.y) }) where !seen.contains(start) {
                var queue = [start]
                seen.insert(start)
                var members: [SIMD3<Int32>] = []
                var head = 0
                while head < queue.count {
                    let c = queue[head]
                    head += 1
                    members.append(contentsOf: cells[c] ?? [])
                    for dx in Int32(-1)...1 {
                        for dy in Int32(-1)...1 where dx != 0 || dy != 0 {
                            let n = SIMD2(c.x + dx, c.y + dy)
                            if cells[n] != nil && !seen.contains(n) {
                                seen.insert(n)
                                queue.append(n)
                            }
                        }
                    }
                }
                guard members.count >= minVoxels else { continue }
                var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
                for v in members {
                    let p = SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z)) * voxel
                    lo = simd_min(lo, p)
                    hi = simd_max(hi, p + SIMD3(repeating: voxel))
                }
                let anchor = members.min { ($0.x, $0.z, $0.y) < ($1.x, $1.z, $1.y) }!
                map.objects.append(ObjectBox(id: "\(classId)@\(anchor.x),\(anchor.y),\(anchor.z)", classId: classId,
                                             label: info.label, group: info.group, min: lo, max: hi,
                                             points: members.count))
            }
        }
        map.objects.sort { $0.points > $1.points }
        return map
    }
}

private func < (a: (Int32, Int32), b: (Int32, Int32)) -> Bool {
    a.0 != b.0 ? a.0 < b.0 : a.1 < b.1
}

private func < (a: (Int32, Int32, Int32), b: (Int32, Int32, Int32)) -> Bool {
    a.0 != b.0 ? a.0 < b.0 : a.1 != b.1 ? a.1 < b.1 : a.2 < b.2
}
