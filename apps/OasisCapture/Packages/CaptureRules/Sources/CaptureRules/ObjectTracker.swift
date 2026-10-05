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
/// joins the tracked object of the same kin whose footprint it overlaps most
/// at a similar height (a cabinet above a desk is not the desk), or starts a
/// new one; a part of a bed or a wardrobe that adjoins one already tracked
/// joins it too, since those are seen in parts (a door, one end). An object
/// remembers every voxel its observations covered, with a hit count, so its
/// box is the extent of everything seen of it from every angle, not of the
/// current view; a voxel needs two hits once the object is established (three
/// when well established), which drops the stray points of a bad depth frame
/// or a mask that was wrong for a moment, and the extent on each axis
/// runs out from the busiest part over everything connected to it (see
/// hitBounds), so a bed is as long as all of it that was seen, not just the
/// side seen most. Labels are votes weighted by confidence, and the shown
/// label only changes with a clear lead. Boxes ease towards new measurements.
/// An object shows after confirmObservations, is forgotten if it never
/// confirms, and once confirmed stays as long as it is out of view; only when
/// the camera looks straight at where it should be, from a sensible distance,
/// and does not find it for forgetObservations frames does it go.
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
            var family: String
        }
        var prepared: [Prepared] = []
        for (i, o) in observations.enumerated() {
            guard let info = spec.info(o.classIndex), info.group != "person", o.points.count >= t.minPoints else { continue }
            var voxels = Set<SIMD3<Int32>>()
            for p in o.points { voxels.insert(Self.key(p, v)) }
            guard voxels.count >= 4,
                  let bounds = Self.hitBounds(voxels.map { (Self.centre($0, v), 1) }, yaw: 0, voxel: v, binShare: t.binShare, gap: t.gapMetres) else { continue }
            let size = bounds.hi - bounds.lo
            // Bigger than any piece of furniture: a wall or floor with a wrong label.
            guard size.x <= t.maxSizeMetres, size.y <= t.maxSizeMetres, size.z <= t.maxSizeMetres else { continue }
            prepared.append(Prepared(index: i, voxels: voxels, min: bounds.lo, max: bounds.hi, kin: spec.kinGroup(of: info.family), family: info.family))
        }

        // Match observations to tracks, best overlap first, one observation per track.
        struct Pair { var observation: Int; var track: Int; var score: Float }
        var pairs: [Pair] = []
        for (pi, p) in prepared.enumerated() {
            let (pMin, pMax) = Self.thickened(p.min, p.max)
            for (ti, track) in tracks.enumerated() {
                guard spec.kinGroup(track.classIndex) == p.kin, let box = track.measured,
                      Self.heightsNear(box.min.y, box.max.y, p.min.y, p.max.y) else { continue }
                let (bMin, bMax) = Self.thickened(box.min, box.max)
                let overlap = RoomMapBuilder.footprintOverlap(bMin, bMax, pMin, pMax)
                // Beds and storage are seen in parts: a part that adjoins one of its own family joins it.
                let reach: Float = Self.seenInParts.contains(p.family) && box.family == p.family ? t.adjoinMetres : 0
                if overlap >= t.matchOverlap {
                    pairs.append(Pair(observation: pi, track: ti, score: overlap))
                } else if reach > 0, RoomMapBuilder.footprintOverlap(box.min, box.max, p.min - SIMD3(reach, 0, reach), p.max + SIMD3(reach, 0, reach)) > 0 {
                    pairs.append(Pair(observation: pi, track: ti, score: 0.01))
                }
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
        // Established objects ignore voxels hit only once or twice (a wrong mask lasts a frame
        // or two; what is there is seen every time it is looked at); young ones cannot afford to.
        var minHits = track.observations >= 30 ? 3 : track.observations >= 3 ? 2 : 1
        var box: ObjectBox?
        while minHits <= 4 {
            let hits = track.voxels.filter { $0.value >= minHits }.map { (Self.centre($0.key, v), $0.value) }
            guard hits.count >= 4, let (lo, hi) = Self.hitBounds(hits, yaw: yaw, voxel: v, binShare: t.binShare, gap: t.gapMetres) else { break }
            let size = hi - lo
            if size.x <= t.maxSizeMetres && size.y <= t.maxSizeMetres && size.z <= t.maxSizeMetres {
                let mid = (lo + hi) / 2
                let ax = SIMD2(cos(yaw), sin(yaw)), az = SIMD2(-sin(yaw), cos(yaw))
                let flat = ax * mid.x + az * mid.z
                guard let info = spec.info(track.classIndex) else { return }
                box = ObjectBox(id: track.id, classId: track.classIndex, label: info.label, group: info.group,
                                family: info.family, center: SIMD3(flat.x, mid.y, flat.y), size: size, yaw: yaw,
                                points: hits.count)
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

    /// Families whose objects are seen in parts (a door at a time, one end of a bed).
    static let seenInParts: Set<String> = ["bed", "storage"]

    /// Bounds in the frame turned by `yaw` from hits per voxel centre: on
    /// each axis the hits are binned at the voxel size, and the extent runs
    /// outward from the busiest bin over every bin with at least `binShare`
    /// of the typical (median) bin's hits, across gaps of up to `gap` metres.
    /// Left out: bins far emptier than the rest, and the wall behind a mask's
    /// edge, which lies beyond a gap; kept: everything connected to the
    /// object, like the side of a bed, though its top (one bin of height)
    /// holds a hundred times the hits. Nil without hits.
    static func hitBounds(_ hits: [(centre: SIMD3<Float>, hits: Int)], yaw: Float, voxel v: Float, binShare: Float,
                          gap: Float) -> (lo: SIMD3<Float>, hi: SIMD3<Float>)? {
        guard !hits.isEmpty else { return nil }
        let ax = SIMD2(cos(yaw), sin(yaw)), az = SIMD2(-sin(yaw), cos(yaw))
        var bins: [[Int: Int]] = [[:], [:], [:]]
        for (c, n) in hits {
            let flat = SIMD2(c.x, c.z)
            let coords = [simd_dot(flat, ax), c.y, simd_dot(flat, az)]
            for i in 0..<3 { bins[i][Int((coords[i] / v).rounded(.down)), default: 0] += n }
        }
        let gapBins = Int((gap / v).rounded())
        var lo = SIMD3<Float>(repeating: 0), hi = SIMD3<Float>(repeating: 0)
        for i in 0..<3 {
            let axis = bins[i]
            guard let busiest = axis.max(by: { $0.value < $1.value }) else { return nil }
            let typical = axis.values.sorted()[axis.count / 2]
            let needed = max(1, Int((binShare * Float(typical)).rounded(.up)))
            let minKey = axis.keys.min()!, maxKey = axis.keys.max()!
            var first = busiest.key, last = busiest.key
            var b = last + 1, empty = 0
            while b <= maxKey && empty <= gapBins {
                if (axis[b] ?? 0) >= needed { last = b; empty = 0 } else { empty += 1 }
                b += 1
            }
            b = first - 1; empty = 0
            while b >= minKey && empty <= gapBins {
                if (axis[b] ?? 0) >= needed { first = b; empty = 0 } else { empty += 1 }
                b -= 1
            }
            lo[i] = Float(first) * v
            hi[i] = Float(last + 1) * v
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
                   Self.heightsOverlap(ba, bb),
                   Self.thickOverlap(ba, bb) >= spec.tracker.mergeOverlap || Self.adjoin(ba, bb, within: spec.tracker.adjoinMetres) {
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
        heightsNear(a.min.y, a.max.y, b.min.y, b.max.y)
    }

    /// Two height ranges that overlap or nearly touch (a wardrobe seen top and
    /// bottom, a desk's top and what stands under it), but not a cabinet on
    /// the wall and the furniture under it.
    static func heightsNear(_ aLo: Float, _ aHi: Float, _ bLo: Float, _ bHi: Float) -> Bool {
        min(aHi, bHi) - max(aLo, bLo) > -0.3
    }

    /// Bounds with a footprint at least 0.3 m each way. A screen or a door is
    /// a slab a few centimetres thick, and two sightings of it a hand apart
    /// in depth would otherwise share no floor at all.
    static func thickened(_ lo: SIMD3<Float>, _ hi: SIMD3<Float>, to minimum: Float = 0.3) -> (SIMD3<Float>, SIMD3<Float>) {
        var lo = lo, hi = hi
        for i in [0, 2] where hi[i] - lo[i] < minimum {
            let mid = (lo[i] + hi[i]) / 2
            lo[i] = mid - minimum / 2
            hi[i] = mid + minimum / 2
        }
        return (lo, hi)
    }

    /// Two parts of one bed or one wardrobe: the same family, seen in parts, touching or nearly.
    private static func adjoin(_ a: ObjectBox, _ b: ObjectBox, within reach: Float) -> Bool {
        guard a.family == b.family, seenInParts.contains(a.family) else { return false }
        let grow = SIMD3(reach, 0, reach)
        return RoomMapBuilder.footprintOverlap(a.min - grow, a.max + grow, b.min, b.max) > 0
    }

    private static func thickOverlap(_ a: ObjectBox, _ b: ObjectBox) -> Float {
        let (aMin, aMax) = thickened(a.min, a.max), (bMin, bMax) = thickened(b.min, b.max)
        return RoomMapBuilder.footprintOverlap(aMin, aMax, bMin, bMax)
    }

    private func match(_ track: Track) -> ObservationMatch {
        let info = spec.info(track.classIndex)
        return ObservationMatch(objectID: track.id, label: info?.label ?? "", group: info?.group ?? "",
                                classIndex: track.classIndex,
                                placed: track.observations >= spec.tracker.confirmObservations && info?.boxed == true)
    }
}
