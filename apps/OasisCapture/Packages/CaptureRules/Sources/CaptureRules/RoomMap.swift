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

/// Builds the room map: keeps the tracking's planes, and turns tracked points
/// labelled by the segmentation into furniture boxes.
///
/// Stability is the whole design. Points vote per voxel, and votes are counted
/// by family, so a sofa the model sometimes calls an armchair still lands in
/// one object; a voxel needs a clear majority. Voxels are grouped on a coarse
/// top-down grid into candidate boxes, then matched to the boxes of the last
/// build: a match keeps its identity and its extent moves only part of the way
/// towards the new measurement, a candidate must be seen in confirmBuilds
/// builds in a row before it shows, and a box unseen for up to graceBuilds
/// builds stays put instead of blinking. A box's label is its family's
/// most-voted member, kept until another member leads by a clear margin.
public final class RoomMapBuilder {
    public let spec: DetectionSpec
    /// Classes never boxed: walls, floor and ceiling come from planes.
    public var excludedGroups: Set<String> = ["structure", "person"]

    private var planes: [UUID: PlaneInfo] = [:]
    /// voxel -> family -> class id -> hits
    private var votes: [SIMD3<Int32>: [String: [Int: Int]]] = [:]
    private var tracked: [Tracked] = []
    private var nextID = 1
    private let lock = NSLock()

    private struct Tracked {
        var box: ObjectBox
        var votes: [Int: Int]       // member class -> hits, for the label
        var seen: Int               // consecutive builds it was matched
        var missed: Int             // consecutive builds it was not
        var shown: Bool
    }

    public init(spec: DetectionSpec) {
        self.spec = spec
    }

    public func reset() {
        lock.withLock {
            planes = [:]
            votes = [:]
            tracked = []
        }
    }

    /// Drops the votes but keeps the placed boxes (tests; a new room segment).
    public func forgetVotes() {
        lock.withLock { votes = [:] }
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
        let v = spec.voteVoxelMetres
        let key = SIMD3<Int32>(Int32((point.x / v).rounded(.down)), Int32((point.y / v).rounded(.down)),
                               Int32((point.z / v).rounded(.down)))
        lock.withLock { votes[key, default: [:]][info.familyName, default: [:]][classId, default: 0] += 1 }
    }

    public func add(points: [(SIMD3<Float>, Int)]) {
        for (p, c) in points { add(point: p, classId: c) }
    }

    /// The map as it stands. Call about once a second: each call is one "build".
    public func build() -> RoomMap {
        lock.withLock {
            var map = RoomMap()
            map.planes = planes.values.sorted { $0.id.uuidString < $1.id.uuidString }
            let candidates = candidateBoxes(yaw: map.roomYaw ?? 0)
            track(candidates)
            map.objects = tracked.filter(\.shown).map(\.box).sorted { $0.points > $1.points }
            return map
        }
    }

    // MARK: Candidates from the voxel votes

    private struct Candidate {
        var family: String
        var votes: [Int: Int]
        var center: SIMD3<Float>
        var size: SIMD3<Float>
        var yaw: Float
        var voxels: Int
        var min: SIMD3<Float> { ObjectBox(id: "", classId: 0, label: "", group: "", family: "", center: center, size: size, yaw: yaw, points: 0).min }
        var max: SIMD3<Float> { ObjectBox(id: "", classId: 0, label: "", group: "", family: "", center: center, size: size, yaw: yaw, points: 0).max }
    }

    private func candidateBoxes(yaw: Float) -> [Candidate] {
        let v = spec.voteVoxelMetres
        // Each voxel takes its majority family, if that majority is clear.
        var byFamily: [String: [(SIMD3<Int32>, [Int: Int])]] = [:]
        for (key, families) in votes {
            let totals = families.mapValues { $0.values.reduce(0, +) }
            guard let best = totals.max(by: { $0.value < $1.value }), best.value >= 3,
                  Double(best.value) >= 0.6 * Double(totals.values.reduce(0, +)) else { continue }
            byFamily[best.key, default: []].append((key, families[best.key] ?? [:]))
        }
        let scale = v / spec.groupCellMetres
        var candidates: [Candidate] = []
        for (family, voxels) in byFamily where voxels.count >= spec.minVoxels {
            // Group on a coarse top-down grid, 8-connected, ignoring height so a
            // tall wardrobe seen in pieces stays one object.
            var cells: [SIMD2<Int32>: [(SIMD3<Int32>, [Int: Int])]] = [:]
            for entry in voxels {
                let c = SIMD2<Int32>(Int32((Float(entry.0.x) * scale).rounded(.down)),
                                     Int32((Float(entry.0.z) * scale).rounded(.down)))
                cells[c, default: []].append(entry)
            }
            var seen = Set<SIMD2<Int32>>()
            for start in cells.keys.sorted(by: { ($0.x, $0.y) < ($1.x, $1.y) }) where !seen.contains(start) {
                var queue = [start]
                seen.insert(start)
                var members: [(SIMD3<Int32>, [Int: Int])] = []
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
                guard members.count >= spec.minVoxels else { continue }
                // Bounds in the room's frame (turned by yaw), so the box lies along the walls.
                let ax = SIMD2(cos(yaw), sin(yaw)), az = SIMD2(-sin(yaw), cos(yaw))
                var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
                var memberVotes: [Int: Int] = [:]
                for (key, counts) in members {
                    let p = (SIMD3<Float>(Float(key.x), Float(key.y), Float(key.z)) + 0.5) * v
                    let flat = SIMD2(p.x, p.z)
                    let local = SIMD3(simd_dot(flat, ax), p.y, simd_dot(flat, az))
                    lo = simd_min(lo, local - v / 2)
                    hi = simd_max(hi, local + v / 2)
                    for (cls, n) in counts { memberVotes[cls, default: 0] += n }
                }
                let size = hi - lo
                // Bigger than any piece of furniture: mislabelled wall or floor points.
                guard size.x <= spec.maxObjectMetres, size.y <= spec.maxObjectMetres, size.z <= spec.maxObjectMetres else { continue }
                let mid = (lo + hi) / 2
                let flatCenter = ax * mid.x + az * mid.z
                candidates.append(Candidate(family: family, votes: memberVotes,
                                            center: SIMD3(flatCenter.x, mid.y, flatCenter.y), size: size, yaw: yaw,
                                            voxels: members.count))
            }
        }
        return candidates
    }

    // MARK: Matching candidates to the boxes already placed

    private func track(_ candidates: [Candidate]) {
        var unmatched = candidates
        for i in tracked.indices {
            let box = tracked[i].box
            // The candidate of the same family overlapping this box the most, from above.
            var bestIndex: Int?
            var bestOverlap: Float = 0
            for (j, c) in unmatched.enumerated() where c.family == box.family {
                let overlap = Self.footprintOverlap(box.min, box.max, c.min, c.max)
                if overlap > bestOverlap { bestOverlap = overlap; bestIndex = j }
            }
            guard let j = bestIndex, bestOverlap > 0.3 else {
                tracked[i].seen = 0
                tracked[i].missed += 1
                continue
            }
            let c = unmatched.remove(at: j)
            // Move part of the way: an object grows smoothly as more of it is seen.
            let k: Float = 0.35
            tracked[i].box.center += (c.center - box.center) * k
            tracked[i].box.size += (c.size - box.size) * k
            tracked[i].box.yaw = c.yaw
            tracked[i].box.points = c.voxels
            tracked[i].votes = c.votes
            tracked[i].seen += 1
            tracked[i].missed = 0
            if tracked[i].seen >= spec.confirmBuilds { tracked[i].shown = true }
            relabel(&tracked[i])
        }
        // Forget boxes that stayed unseen too long.
        tracked.removeAll { $0.missed > spec.graceBuilds }
        for c in unmatched {
            let top = c.votes.max { $0.value < $1.value }?.key ?? 0
            guard let info = spec.info(top) else { continue }
            tracked.append(Tracked(
                box: ObjectBox(id: "o\(nextID)", classId: top, label: info.label, group: info.group, family: c.family,
                               center: c.center, size: c.size, yaw: c.yaw, points: c.voxels),
                votes: c.votes, seen: 1, missed: 0, shown: spec.confirmBuilds <= 1))
            nextID += 1
        }
    }

    /// Another member of the family takes the label only with a clear lead.
    private func relabel(_ t: inout Tracked) {
        guard let top = t.votes.max(by: { $0.value < $1.value }) else { return }
        let current = t.votes[t.box.classId] ?? 0
        guard top.key != t.box.classId, Double(top.value) > 1.25 * Double(current) + 2,
              let info = spec.info(top.key) else { return }
        t.box.classId = top.key
        t.box.label = info.label
        t.box.group = info.group
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

private func < (a: (Int32, Int32), b: (Int32, Int32)) -> Bool {
    a.0 != b.0 ? a.0 < b.0 : a.1 < b.1
}
