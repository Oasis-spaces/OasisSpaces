import Foundation
import simd

/// One detected thing in one frame, lifted into the room: the world points
/// its mask covers (depth behind the mask's pixels).
public struct ObjectObservation: Sendable {
    public var classIndex: Int
    public var confidence: Float
    public var points: [SIMD3<Float>]

    public init(classIndex: Int, confidence: Float, points: [SIMD3<Float>]) {
        self.classIndex = classIndex
        self.confidence = confidence
        self.points = points
    }
}

/// Where an observation went: the tracked object it joined (or started).
public struct ObservationMatch: Sendable, Equatable {
    public var objectID: String
    public var label: String
    public var group: String
    public var classIndex: Int
    /// Whether the object is placed in the room map (unconfirmed objects are not yet).
    public var placed: Bool
}

/// Keeps the room's objects across frames.
///
/// Every observation is a cloud of world points with a class. An observation
/// joins the tracked object of the same kin whose footprint it overlaps most,
/// or starts a new one. An object remembers every voxel its observations
/// covered, with a hit count, so its box is the extent of everything seen of
/// it from every angle, not of the current view; a voxel needs two hits once
/// the object is established, which drops the stray points of one bad depth
/// frame. Labels are votes weighted by confidence, and the shown label only
/// changes with a clear lead. Boxes ease towards new measurements. An object
/// shows after confirmObservations, is forgotten if it never confirms, and
/// once confirmed stays as long as it is out of view; only when the camera
/// looks straight at where it should be, from a sensible distance, and does
/// not find it for forgetObservations frames does it go.
public final class ObjectTracker {
    public let spec: ObjectSpec
    private var tracks: [Track] = []
    private var nextID = 1
    private var frame = 0

    struct Track {
        var id: String
        var classIndex: Int
        var votes: [Int: Float]
        var voxels: [SIMD3<Int32>: Int]
        var observations: Int
        var lastSeen: Int
        var missedInView: Int
        var box: ObjectBox?          // eased, nil until first measured
        var measured: ObjectBox?     // the latest extent measurement
        var born: Int
        /// Identities of objects merged into this one, so an observation's match stays valid.
        var mergedIDs: Set<String> = []
    }

    public init(spec: ObjectSpec) {
        self.spec = spec
    }

    public func reset() {
        tracks = []
        frame = 0
    }

    /// The objects placed so far: confirmed, boxed classes, with their eased boxes.
    public var objects: [ObjectBox] {
        tracks.compactMap { t in
            guard t.observations >= spec.tracker.confirmObservations, let box = t.box,
                  spec.info(t.classIndex)?.boxed == true else { return nil }
            return box
        }.sorted { $0.points > $1.points }
    }

    /// Every tracked object, placed or not (for labels on screen).
    public var count: Int { tracks.count }

    /// Feeds one frame's observations. `yaw` is the room's direction (boxes
    /// are turned to it); `camera` is where the frame was taken from, used to
    /// tell "not detected while looked at" from "out of view". Returns a
    /// match per observation, in order.
    @discardableResult
    public func observe(_ observations: [ObjectObservation], yaw: Float, camera: PinholeCamera?) -> [ObservationMatch?] {
        frame += 1
        let t = spec.tracker
        let v = t.voxelMetres
        let near = t.depthMetres.first ?? 0.3, far = t.depthMetres.last ?? 6

        // Each observation's own voxels and extent.
        struct Prepared {
            var index: Int
            var voxels: Set<SIMD3<Int32>>
            var min: SIMD3<Float>
            var max: SIMD3<Float>
            var kin: String
        }
        var prepared: [Prepared] = []
        for (i, o) in observations.enumerated() {
            guard let info = spec.info(o.classIndex), info.group != "person", o.points.count >= t.minPoints else { continue }
            var voxels = Set<SIMD3<Int32>>()
            for p in o.points { voxels.insert(Self.key(p, v)) }
            guard voxels.count >= 4 else { continue }
            let (lo, hi) = Self.trimmedBounds(voxels.map { Self.centre($0, v) }, yaw: 0, share: t.trimShare)
            let size = hi - lo
            // Bigger than any piece of furniture: a wall or floor with a wrong label.
            guard size.x <= t.maxSizeMetres, size.y <= t.maxSizeMetres, size.z <= t.maxSizeMetres else { continue }
            prepared.append(Prepared(index: i, voxels: voxels, min: lo, max: hi, kin: spec.kinGroup(of: info.family)))
        }

        // Match observations to tracks, best overlap first, one observation per track.
        struct Pair { var observation: Int; var track: Int; var score: Float }
        var pairs: [Pair] = []
        for (pi, p) in prepared.enumerated() {
            for (ti, track) in tracks.enumerated() {
                guard spec.kinGroup(track.classIndex) == p.kin, let box = track.measured else { continue }
                let overlap = RoomMapBuilder.footprintOverlap(box.min, box.max, p.min, p.max)
                // Things stacked on each other (a pillow on a bed) are different kin, so
                // height does not need checking; a thin overlap from above is enough.
                if overlap >= t.matchOverlap { pairs.append(Pair(observation: pi, track: ti, score: overlap)) }
            }
        }
        pairs.sort { $0.score > $1.score }
        var matchOf = [Int?](repeating: nil, count: prepared.count)
        var taken = Set<Int>()
        for pair in pairs where matchOf[pair.observation] == nil && !taken.contains(pair.track) {
            matchOf[pair.observation] = pair.track
            taken.insert(pair.track)
        }

        var results = [ObservationMatch?](repeating: nil, count: observations.count)
        for (pi, p) in prepared.enumerated() {
            let o = observations[p.index]
            if let ti = matchOf[pi] {
                for key in p.voxels { tracks[ti].voxels[key, default: 0] += 1 }
                tracks[ti].votes[o.classIndex, default: 0] += o.confidence
                tracks[ti].observations += 1
                tracks[ti].lastSeen = frame
                tracks[ti].missedInView = 0
                relabel(&tracks[ti])
                results[p.index] = match(tracks[ti])
            } else {
                var voxels: [SIMD3<Int32>: Int] = [:]
                for key in p.voxels { voxels[key] = 1 }
                let track = Track(id: "o\(nextID)", classIndex: o.classIndex, votes: [o.classIndex: o.confidence],
                                  voxels: voxels, observations: 1, lastSeen: frame, missedInView: 0,
                                  box: nil, measured: nil, born: frame)
                nextID += 1
                tracks.append(track)
                results[p.index] = match(track)
            }
        }

        // Misses: a track the camera should be seeing, but no observation joined.
        for i in tracks.indices where tracks[i].lastSeen != frame {
            guard let camera, let box = tracks[i].measured, let (u, vv, z) = camera.project(box.center),
                  z >= max(near, 0.8), z <= far,
                  u > Float(camera.width) * 0.15, u < Float(camera.width) * 0.85,
                  vv > Float(camera.height) * 0.15, vv < Float(camera.height) * 0.85 else { continue }
            tracks[i].missedInView += 1
        }
        tracks.removeAll { track in
            let confirmed = track.observations >= t.confirmObservations
            if !confirmed { return frame - track.lastSeen > t.staleObservations }
            return track.missedInView >= t.forgetObservations
        }

        // Measure and ease every track touched this frame (and any not yet measured).
        for i in tracks.indices where tracks[i].lastSeen == frame || tracks[i].measured == nil {
            measure(&tracks[i], yaw: yaw)
        }
        merge(yaw: yaw)
        // Merged-away tracks may have carried a match; point those at the survivor.
        for i in results.indices {
            guard let r = results[i], !tracks.contains(where: { $0.id == r.objectID }),
                  let survivor = tracks.first(where: { $0.mergedIDs.contains(r.objectID) }) else { continue }
            results[i] = match(survivor)
        }
        return results
    }

    /// Re-measures every box against a new room direction (walls found later).
    public func reorient(yaw: Float) {
        for i in tracks.indices { measure(&tracks[i], yaw: yaw, jump: true) }
    }

    // MARK: Measuring

    private func measure(_ track: inout Track, yaw: Float, jump: Bool = false) {
        let t = spec.tracker
        let v = t.voxelMetres
        // Established objects ignore voxels hit only once; young ones cannot afford to.
        var minHits = track.observations >= 3 ? 2 : 1
        var centres: [SIMD3<Float>] = []
        var box: ObjectBox?
        while minHits <= 4 {
            centres = track.voxels.filter { $0.value >= minHits }.map { Self.centre($0.key, v) }
            guard centres.count >= 4 else { break }
            let (lo, hi) = Self.trimmedBounds(centres, yaw: yaw, share: t.trimShare)
            let size = simd_max(hi - lo, SIMD3(repeating: v))
            if size.x <= t.maxSizeMetres && size.y <= t.maxSizeMetres && size.z <= t.maxSizeMetres {
                let mid = (lo + hi) / 2
                let ax = SIMD2(cos(yaw), sin(yaw)), az = SIMD2(-sin(yaw), cos(yaw))
                let flat = ax * mid.x + az * mid.z
                guard let info = spec.info(track.classIndex) else { return }
                box = ObjectBox(id: track.id, classId: track.classIndex, label: info.label, group: info.group,
                                family: info.family, center: SIMD3(flat.x, mid.y, flat.y), size: size, yaw: yaw,
                                points: centres.count)
                break
            }
            minHits += 1   // too big: demand more agreement
        }
        guard let measured = box else { return }
        track.measured = measured
        if var eased = track.box, !jump {
            let k = t.easing
            eased.center += (measured.center - eased.center) * k
            eased.size += (measured.size - eased.size) * k
            eased.yaw = measured.yaw
            eased.points = measured.points
            eased.classId = measured.classId
            eased.label = measured.label
            eased.group = measured.group
            eased.family = measured.family
            track.box = eased
        } else {
            track.box = measured
        }
    }

    /// Bounds in the frame turned by `yaw`, ignoring outliers: on each axis
    /// the points more than 3.5 median absolute deviations from the median
    /// (the edge pixels of a mask land on the wall behind, metres away), then
    /// `share` of what is left at each end.
    static func trimmedBounds(_ points: [SIMD3<Float>], yaw: Float, share: Float) -> (SIMD3<Float>, SIMD3<Float>) {
        let ax = SIMD2(cos(yaw), sin(yaw)), az = SIMD2(-sin(yaw), cos(yaw))
        var axes: [[Float]] = [[], [], []]
        for i in 0..<3 { axes[i].reserveCapacity(points.count) }
        for p in points {
            let flat = SIMD2(p.x, p.z)
            axes[0].append(simd_dot(flat, ax)); axes[1].append(p.y); axes[2].append(simd_dot(flat, az))
        }
        var lo = SIMD3<Float>(repeating: 0), hi = SIMD3<Float>(repeating: 0)
        for i in 0..<3 {
            let sorted = axes[i].sorted()
            let n = sorted.count
            let median = sorted[n / 2]
            let mad = axes[i].map { abs($0 - median) }.sorted()[n / 2]
            let reach = max(3.5 * mad, 0.05)
            let kept = sorted.filter { abs($0 - median) <= reach }
            let drop = min(max(0, kept.count / 2 - 1), Int((Float(kept.count) * share).rounded()))
            lo[i] = kept[drop]
            hi[i] = kept[kept.count - 1 - drop]
        }
        return (lo, hi)
    }

    static func key(_ p: SIMD3<Float>, _ v: Float) -> SIMD3<Int32> {
        SIMD3(Int32((p.x / v).rounded(.down)), Int32((p.y / v).rounded(.down)), Int32((p.z / v).rounded(.down)))
    }

    static func centre(_ k: SIMD3<Int32>, _ v: Float) -> SIMD3<Float> {
        (SIMD3<Float>(Float(k.x), Float(k.y), Float(k.z)) + 0.5) * v
    }

    // MARK: Labels and merging

    /// Another class takes the label only with a clear lead in the votes.
    private func relabel(_ track: inout Track) {
        guard let top = track.votes.max(by: { $0.value < $1.value }), top.key != track.classIndex else { return }
        let current = track.votes[track.classIndex] ?? 0
        if top.value > 1.25 * current + 1.0 { track.classIndex = top.key }
    }

    /// Two objects of one kin whose footprints mostly overlap are one object
    /// seen from two sides before the sides joined up.
    private func merge(yaw: Float) {
        var i = 0
        while i < tracks.count {
            var j = i + 1
            var mergedAny = false
            while j < tracks.count {
                let a = tracks[i], b = tracks[j]
                if spec.kinGroup(a.classIndex) == spec.kinGroup(b.classIndex), let ba = a.measured, let bb = b.measured,
                   RoomMapBuilder.footprintOverlap(ba.min, ba.max, bb.min, bb.max) >= spec.tracker.mergeOverlap,
                   Self.heightsOverlap(ba, bb) {
                    // The older keeps its identity.
                    let (keep, drop) = a.born <= b.born ? (i, j) : (j, i)
                    var survivor = tracks[keep]
                    let gone = tracks[drop]
                    for (k, n) in gone.voxels { survivor.voxels[k, default: 0] += n }
                    for (c, w) in gone.votes { survivor.votes[c, default: 0] += w }
                    survivor.observations += gone.observations
                    survivor.lastSeen = max(survivor.lastSeen, gone.lastSeen)
                    survivor.mergedIDs.insert(gone.id)
                    survivor.mergedIDs.formUnion(gone.mergedIDs)
                    relabel(&survivor)
                    tracks[keep] = survivor
                    tracks.remove(at: drop)
                    measure(&tracks[keep < drop ? keep : keep - 1], yaw: yaw)
                    mergedAny = true
                    break
                }
                j += 1
            }
            if !mergedAny { i += 1 }
        }
    }

    private static func heightsOverlap(_ a: ObjectBox, _ b: ObjectBox) -> Bool {
        let lo = max(a.center.y - a.size.y / 2, b.center.y - b.size.y / 2)
        let hi = min(a.center.y + a.size.y / 2, b.center.y + b.size.y / 2)
        return hi - lo > -0.3   // touching or nearly so (a wardrobe seen top and bottom)
    }

    private func match(_ track: Track) -> ObservationMatch {
        let info = spec.info(track.classIndex)
        return ObservationMatch(objectID: track.id, label: info?.label ?? "", group: info?.group ?? "",
                                classIndex: track.classIndex,
                                placed: track.observations >= spec.tracker.confirmObservations && info?.boxed == true)
    }
}
