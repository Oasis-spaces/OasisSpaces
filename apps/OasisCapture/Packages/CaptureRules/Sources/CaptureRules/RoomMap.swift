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

/// A piece of furniture (or fixture) the scan has placed: a box turned by
/// `yaw` about the vertical, so it lies along the room's walls like the
/// furniture does.
public struct ObjectBox: Identifiable, Sendable, Equatable {
    public var id: String
    public var classId: Int
    public var label: String
    public var group: String
    public var family: String
    public var center: SIMD3<Float>
    /// Width (along the box's own x), height, depth (along its own z), metres.
    public var size: SIMD3<Float>
    /// Turn about the vertical, radians, of the box's own x axis from world x.
    public var yaw: Float
    public var points: Int

    public init(id: String, classId: Int, label: String, group: String, family: String,
                center: SIMD3<Float>, size: SIMD3<Float>, yaw: Float, points: Int) {
        self.id = id; self.classId = classId; self.label = label; self.group = group; self.family = family
        self.center = center; self.size = size; self.yaw = yaw; self.points = points
    }

    /// The box's own axes on the floor (x, z).
    public var axisX: SIMD2<Float> { SIMD2(cos(yaw), sin(yaw)) }
    public var axisZ: SIMD2<Float> { SIMD2(-sin(yaw), cos(yaw)) }

    /// The four corners of the footprint, seen from above (x, z), in order.
    public var footprint: [SIMD2<Float>] {
        let c = SIMD2(center.x, center.z)
        let hx = axisX * (size.x / 2), hz = axisZ * (size.z / 2)
        return [c - hx - hz, c + hx - hz, c + hx + hz, c - hx + hz]
    }

    /// World-aligned bounds of the whole box.
    public var min: SIMD3<Float> {
        let f = footprint
        return SIMD3(f.map(\.x).min()!, center.y - size.y / 2, f.map(\.y).min()!)
    }

    public var max: SIMD3<Float> {
        let f = footprint
        return SIMD3(f.map(\.x).max()!, center.y + size.y / 2, f.map(\.y).max()!)
    }

    public var footprintMin: SIMD2<Float> { SIMD2(min.x, min.z) }
    public var footprintMax: SIMD2<Float> { SIMD2(max.x, max.z) }
}

/// What has been detected in the room so far, for the map and the overlay.
public struct RoomMap: Sendable, Equatable {
    public var planes: [PlaneInfo] = []
    public var objects: [ObjectBox] = []

    public init() {}

    public init(planes: [PlaneInfo], objects: [ObjectBox] = []) {
        self.planes = planes
        self.objects = objects
    }

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

    /// The direction the room's walls run, radians in 0..<pi/2: the longest
    /// walls' directions folded into one quarter turn and averaged. Furniture
    /// boxes are turned to it. Nil without walls.
    public var roomYaw: Float? {
        let walls = self.walls.filter { simd_distance($0.from, $0.to) > 0.5 }
        guard !walls.isEmpty else { return nil }
        // Average the doubled-doubled angle, so directions a quarter turn apart agree.
        var sx: Float = 0, sy: Float = 0
        for wall in walls {
            let d = wall.to - wall.from
            let angle = atan2(d.y, d.x) * 4
            let weight = simd_length(d)
            sx += cos(angle) * weight
            sy += sin(angle) * weight
        }
        var yaw = atan2(sy, sx) / 4
        if yaw < 0 { yaw += .pi / 2 }
        return yaw
    }

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

/// Builds the room map: keeps the tracking's planes (walls, floor) and the
/// objects the tracker has placed from the detector's observations.
public final class RoomMapBuilder {
    public let spec: ObjectSpec
    public let tracker: ObjectTracker

    private var planes: [UUID: PlaneInfo] = [:]
    private var lastYaw: Float?
    private let lock = NSLock()

    public init(spec: ObjectSpec) {
        self.spec = spec
        tracker = ObjectTracker(spec: spec)
    }

    public func reset() {
        lock.withLock {
            planes = [:]
            tracker.reset()
            lastYaw = nil
        }
    }

    public func update(plane: PlaneInfo) {
        lock.withLock { planes[plane.id] = plane }
    }

    public func remove(plane id: UUID) {
        lock.withLock { planes[id] = nil }
    }

    /// The direction the walls run, as known now.
    public var roomYaw: Float? {
        lock.withLock { RoomMap(planes: Array(planes.values)).roomYaw }
    }

    /// One frame's detections, lifted into the room. Returns where each went.
    @discardableResult
    public func observe(_ observations: [ObjectObservation], camera: PinholeCamera?) -> [ObservationMatch?] {
        lock.withLock {
            let yaw = RoomMap(planes: Array(planes.values)).roomYaw ?? 0
            // Walls found later turn every box already placed.
            if let last = lastYaw, abs(last - yaw) > 0.05 { tracker.reorient(yaw: yaw) }
            lastYaw = yaw
            return tracker.observe(observations, yaw: yaw, camera: camera)
        }
    }

    /// The map as it stands.
    public func build() -> RoomMap {
        lock.withLock {
            var map = RoomMap()
            map.planes = planes.values.sorted { $0.id.uuidString < $1.id.uuidString }
            map.objects = tracker.objects
            return map
        }
    }

    /// Share of the smaller footprint the two boxes share, seen from above.
    static func footprintOverlap(_ aMin: SIMD3<Float>, _ aMax: SIMD3<Float>, _ bMin: SIMD3<Float>, _ bMax: SIMD3<Float>) -> Float {
        let w = Swift.max(0, Swift.min(aMax.x, bMax.x) - Swift.max(aMin.x, bMin.x))
        let d = Swift.max(0, Swift.min(aMax.z, bMax.z) - Swift.max(aMin.z, bMin.z))
        let areaA = (aMax.x - aMin.x) * (aMax.z - aMin.z), areaB = (bMax.x - bMin.x) * (bMax.z - bMin.z)
        let smaller = Swift.max(1e-6, Swift.min(areaA, areaB))
        return w * d / smaller
    }
}
