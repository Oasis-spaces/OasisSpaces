import Foundation
import simd

/// One detected thing in one frame, lifted into the room: the world points
/// its mask covers (depth behind the mask's pixels).
public struct ObjectObservation: Sendable {
    public var classIndex: Int
    public var confidence: Float
    public var points: [SIMD3<Float>]
    /// The surface model saw bare wall where this is (see ObjectSpec.isOnBareSurface):
    /// it is not a thing of its own, but it may be a door of a wardrobe already tracked.
    public var doubtful: Bool

    public init(classIndex: Int, confidence: Float, points: [SIMD3<Float>], doubtful: Bool = false) {
        self.classIndex = classIndex
        self.confidence = confidence
        self.points = points
        self.doubtful = doubtful
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
    /// How many times the object has been seen, this time included.
    public var sightings: Int
}

/// Keeps the room's objects across frames.
///
/// Every observation is a cloud of world points with a class. An observation
/// joins the tracked object of the same kin whose footprint it overlaps most
/// at a similar height (a cabinet above a desk is not the desk), or starts a
/// new one; a part of a bed or a wardrobe that adjoins one already tracked
/// joins it too, since those are seen in parts (a door, one end). A curtain,
/// window or mirror that lies in the plane of a standing wardrobe's front and
/// adjoins it is one of the wardrobe's doors (a patterned sliding door is a
/// curtain to the detector) and becomes part of the wardrobe, whichever was
/// seen first. And what is seen again from the same standpoint in the same
/// direction is the same thing, wherever that frame's depth put it: turning
/// on the spot gives the depth nothing to hold on to, it jumps from frame to
/// frame, and one monitor became three in a row along the line of sight. Its
/// points join the object all the same; the box is of where it was seen
/// most (the busiest connected part), so the stray depths fall away. An object
/// remembers every voxel its observations covered, with a hit count, so its
/// box is the extent of everything seen of it from every angle, not of the
/// current view; once the object is established a voxel needs two hits (a
/// hit next to a cell already seen counts for that cell: depth jitters by
/// about a cell between frames), and more than one glimpse of it (seen at
/// length, or again later), which drops
/// the stray points of a bad depth frame and a mask that was wrong for a
/// moment, and the extent on each axis
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
    /// Told of every merge, for following what the tracker did (the simulator's --trace).
    public var trace: ((String) -> Void)?
    private var tracks: [Track] = []
    private var nextID = 1
    private var frame = 0
    /// Height of the floor, when the room map knows it.
    private var floor: Float?

    struct Track {
        var id: String
        var classIndex: Int
        var votes: [Int: Float]
        var voxels: [SIMD3<Int32>: Hits]
        var observations: Int
        var lastSeen: Int
        var missedInView: Int
        var box: ObjectBox?          // eased, nil until first measured
        var measured: ObjectBox?     // the latest extent measurement
        var born: Int
        /// Identities of objects merged into this one, so an observation's match stays valid.
        var mergedIDs: Set<String> = []
        /// The latest measurement in the room's own axes (see Extent): of everything
        /// seen of it, which is what a part or a door is held against.
        var extent: Extent?
        /// Made from a doubtful observation: it lasts only if it turns out to be a wardrobe's door.
        var doubtful = false
        /// Where it was last seen from, and where its sighting's middle was (see sameSight).
        var seenFrom: SIMD3<Float>?
        var seenAt = SIMD3<Float>(repeating: 0)

        /// One more sighting of the voxel at `key`. Depth jitters by about a voxel from
        /// frame to frame, so the same patch of a surface seldom lands in the same 5 cm
        /// cell twice: a hit on a new cell next to one seen in an earlier frame counts for
        /// that one, and "seen twice" means seen twice within a cell of itself. (Only an
        /// earlier frame's cell: the new cells of one sighting must not vouch for each other.)
        mutating func hit(_ key: SIMD3<Int32>, frame: Int) {
            let one = Hits(count: 1, first: frame, last: frame)
            if voxels[key] != nil {
                voxels[key]!.add(one)
                return
            }
            for step in [SIMD3<Int32>(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)] {
                let neighbour = key &+ step
                if let seen = voxels[neighbour], seen.last < frame {
                    voxels[neighbour]!.add(one)
                    return
                }
            }
            voxels[key] = one
        }
    }

    /// How often a voxel was seen as part of an object, and over which frames.
    struct Hits {
        var count: Int
        var first: Int
        var last: Int
        /// Vouched for otherwise than by being seen at length (a door the unit took).
        var vouched = false

        mutating func add(_ other: Hits) {
            count += other.count
            first = min(first, other.first)
            last = max(last, other.last)
            vouched = vouched || other.vouched
        }
    }

    /// An object's extent in the frame turned by the room's yaw (x along the
    /// walls, y up, z across): its bounds, where it is busiest on each axis
    /// (for a wardrobe seen from the front, the plane of its doors), and its
    /// top layer (the highest level with a good share of the hits: a table's top).
    struct Extent {
        var lo: SIMD3<Float>
        var hi: SIMD3<Float>
        var busiest: SIMD3<Float>
        var top: Float
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
            guard !t.doubtful, t.observations >= spec.tracker.confirmObservations, let box = t.box,
                  spec.info(t.classIndex)?.boxed == true else { return nil }
            return box
        }.sorted { $0.points > $1.points }
    }

    /// Every tracked object, placed or not (for labels on screen).
    public var count: Int { tracks.count }

    /// Feeds one frame's observations. `yaw` is the room's direction (boxes
    /// are turned to it); `floor` its floor's height, when known (a standing
    /// piece's footprint is its lower body's); `camera` is where the frame was taken from, used to
    /// tell "not detected while looked at" from "out of view". Returns a
    /// match per observation, in order.
    @discardableResult
    public func observe(_ observations: [ObjectObservation], yaw: Float, floor: Float? = nil, camera: PinholeCamera?) -> [ObservationMatch?] {
        frame += 1
        self.floor = floor
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
            var doubtful: Bool
            /// For a part of a storage unit: its extent in the room's axes (is it in the unit's plane?).
            var part: Extent?
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
            let part = info.family == "storage" && !o.doubtful
                ? Self.extent(voxels.map { (Self.centre($0, v), 1) }, yaw: yaw, voxel: v, binShare: t.binShare, gap: t.gapMetres) : nil
            prepared.append(Prepared(index: i, voxels: voxels, min: bounds.lo, max: bounds.hi, kin: spec.kinGroup(of: info.family),
                                     family: info.family, doubtful: o.doubtful, part: part))
        }

        // Match observations to tracks, best overlap first, one observation per track.
        struct Pair { var observation: Int; var track: Int; var score: Float }
        var pairs: [Pair] = []
        for (pi, p) in prepared.enumerated() where !p.doubtful {
            let (pMin, pMax) = Self.thickened(p.min, p.max)
            for (ti, track) in tracks.enumerated() {
                guard spec.kinGroup(track.classIndex) == p.kin, let box = track.measured,
                      Self.heightsNear(box.min.y, box.max.y, p.min.y, p.max.y) else { continue }
                let (bMin, bMax) = Self.thickened(box.min, box.max)
                let overlap = RoomMapBuilder.footprintOverlap(bMin, bMax, pMin, pMax)
                // Beds and storage are seen in parts: a part that adjoins one of its own family joins it
                // (for storage, in the same plane: two units that meet in a corner of the room are two).
                var reach: Float = Self.seenInParts.contains(p.family) && box.family == p.family ? t.adjoinMetres : 0
                if let part = p.part, let unit = track.extent, !Self.coplanar(part, unit) { reach = 0 }
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
        // What found no object where its depth put it may be one seen from here before, in this direction.
        if let eye = camera?.position, t.standpointMetres > 0 {
            var sights: [Pair] = []
            for (pi, p) in prepared.enumerated() where matchOf[pi] == nil && !p.doubtful {
                let middle = (p.min + p.max) / 2
                let radius = simd_length(p.max - p.min) / 2
                for (ti, track) in tracks.enumerated() where !taken.contains(ti) && spec.kinGroup(track.classIndex) == p.kin {
                    if let off = Self.sameSight(from: eye, at: middle, radius: radius, track: track, within: t.standpointMetres) {
                        sights.append(Pair(observation: pi, track: ti, score: -off))
                    }
                }
            }
            sights.sort { $0.score > $1.score }
            for pair in sights where matchOf[pair.observation] == nil && !taken.contains(pair.track) {
                matchOf[pair.observation] = pair.track
                taken.insert(pair.track)
            }
        }

        var results = [ObservationMatch?](repeating: nil, count: observations.count)
        for (pi, p) in prepared.enumerated() {
            let o = observations[p.index]
            if let ti = matchOf[pi] {
                for key in p.voxels { tracks[ti].hit(key, frame: frame) }
                tracks[ti].votes[o.classIndex, default: 0] += o.confidence
                tracks[ti].observations += 1
                tracks[ti].lastSeen = frame
                tracks[ti].missedInView = 0
                tracks[ti].seenFrom = camera?.position
                tracks[ti].seenAt = (p.min + p.max) / 2
                relabel(&tracks[ti])
                results[p.index] = match(tracks[ti])
            } else {
                var voxels: [SIMD3<Int32>: Hits] = [:]
                for key in p.voxels { voxels[key] = Hits(count: 1, first: frame, last: frame) }
                // (A doubtful observation joins nothing: it stands alone until the merge below
                // finds it to be a wardrobe's door, or it is dropped at the end of this frame.)
                var track = Track(id: "o\(nextID)", classIndex: o.classIndex, votes: [o.classIndex: o.confidence],
                                  voxels: voxels, observations: 1, lastSeen: frame, missedInView: 0,
                                  box: nil, measured: nil, born: frame)
                track.doubtful = p.doubtful
                track.seenFrom = camera?.position
                track.seenAt = (p.min + p.max) / 2
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
        // What was doubtful and turned out to be nobody's door was bare wall with a name.
        tracks.removeAll { $0.doubtful }
        // Merged-away tracks may have carried a match; point those at the survivor.
        for i in results.indices {
            guard let r = results[i], !tracks.contains(where: { $0.id == r.objectID }) else { continue }
            results[i] = tracks.first { $0.mergedIDs.contains(r.objectID) }.map(match)
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
        // Established objects ignore voxels hit only once (the stray points of one bad depth
        // frame), and what was seen only in one glimpse: a mask that was wrong for a moment (a
        // door called a wardrobe) is wrong for two or three frames running and then never again,
        // while what is there is seen at length, or again on a later pass, or most of the times
        // the object is. Young objects cannot afford either rule.
        let established = track.observations >= 3
        var minHits = established ? 2 : 1
        var box: ObjectBox?
        while minHits <= 4 {
            let hits = track.voxels.filter { _, h in
                h.count >= minHits && (!established || h.vouched || h.last - h.first >= t.glimpseAnalyses || h.count >= 2 * minHits
                                       || 2 * h.count >= track.observations)
            }.map { (Self.centre($0.key, v), $0.value.count) }
            guard hits.count >= 4, let extent = Self.extent(hits, yaw: yaw, voxel: v, binShare: t.binShare, gap: t.gapMetres) else { break }
            // What stands on the floor covers the floor its lower body covers: cabinets that run on
            // over a doorway at head height are part of the wardrobe, not of its footprint.
            var lo = extent.lo, hi = extent.hi
            if let floor, extent.lo.y - floor <= t.snapMetres {
                let lower = hits.filter { $0.0.y < floor + t.bodyMetres }
                if lower.count >= max(4, hits.count / 6),
                   let body = Self.extent(lower, yaw: yaw, voxel: v, binShare: t.binShare, gap: t.gapMetres) {
                    lo.x = body.lo.x; hi.x = body.hi.x
                    lo.z = body.lo.z; hi.z = body.hi.z
                }
            }
            let size = hi - lo
            if size.x <= t.maxSizeMetres && size.y <= t.maxSizeMetres && size.z <= t.maxSizeMetres {
                let mid = (lo + hi) / 2
                let ax = SIMD2(cos(yaw), sin(yaw)), az = SIMD2(-sin(yaw), cos(yaw))
                let flat = ax * mid.x + az * mid.z
                guard let info = spec.info(track.classIndex) else { return }
                box = ObjectBox(id: track.id, classId: track.classIndex, label: info.label, group: info.group,
                                family: info.family, center: SIMD3(flat.x, mid.y, flat.y), size: size, yaw: yaw,
                                points: hits.count, top: extent.top)
                track.extent = extent
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

    /// Whether something seen from `eye` with its middle at `middle` is the
    /// track seen before: from the same standpoint (within a step) the two
    /// lie in the same direction (within half the thing's apparent size, or
    /// four degrees). Returns how far off the directions are, radians.
    static func sameSight(from eye: SIMD3<Float>, at middle: SIMD3<Float>, radius: Float, track: Track, within step: Float) -> Float? {
        guard let from = track.seenFrom, simd_distance(eye, from) <= step else { return nil }
        let now = middle - eye, then = track.seenAt - from
        let far = simd_length(now), was = simd_length(then)
        guard far > 0.2, was > 0.2 else { return nil }
        let off = acos(max(-1, min(1, simd_dot(now, then) / (far * was))))
        return off <= max(4 * .pi / 180, 0.5 * atan(radius / far)) ? off : nil
    }

    /// Flat things that can be a wardrobe's door: what the detector calls a
    /// patterned or mirrored door. (Not "door": a room's door beside a fitted
    /// wardrobe lies in the same plane and is not part of it.)
    static let panelFamilies: Set<String> = ["curtain", "window", "mirror"]

    /// Whether a flat thing is a door of the tracked unit: no thicker than a
    /// door, in the plane where the unit is busiest (its front, give or take
    /// what depth differs by between two passes), lying partly over the unit
    /// along that plane, at its height. A real curtain beside a wardrobe hangs
    /// at the wall, a wardrobe's depth behind its front; a cloth hung next to
    /// it does not lie over it.
    static func isDoor(_ panel: Extent, of unit: Extent) -> Bool {
        let size = panel.hi - panel.lo
        let across = size.x <= size.z ? 0 : 2, along = 2 - across
        guard size[across] <= 0.3 else { return false }
        let plane = (panel.lo[across] + panel.hi[across]) / 2
        guard abs(plane - unit.busiest[across]) <= 0.25 else { return false }
        let shared = min(panel.hi[along], unit.hi[along]) - max(panel.lo[along], unit.lo[along])
        return shared >= max(0.15, 0.2 * size[along]) && heightsNear(unit.lo.y, unit.hi.y, panel.lo.y, panel.hi.y)
    }

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
        extent(hits, yaw: yaw, voxel: v, binShare: binShare, gap: gap).map { ($0.lo, $0.hi) }
    }

    /// The same, with where the object is busiest on each axis and its top layer.
    static func extent(_ hits: [(centre: SIMD3<Float>, hits: Int)], yaw: Float, voxel v: Float, binShare: Float, gap: Float) -> Extent? {
        guard !hits.isEmpty else { return nil }
        let ax = SIMD2(cos(yaw), sin(yaw)), az = SIMD2(-sin(yaw), cos(yaw))
        var bins: [[Int: Int]] = [[:], [:], [:]]
        for (c, n) in hits {
            let flat = SIMD2(c.x, c.z)
            let coords = [simd_dot(flat, ax), c.y, simd_dot(flat, az)]
            for i in 0..<3 { bins[i][Int((coords[i] / v).rounded(.down)), default: 0] += n }
        }
        let gapBins = Int((gap / v).rounded())
        var lo = SIMD3<Float>(repeating: 0), hi = SIMD3<Float>(repeating: 0), busy = SIMD3<Float>(repeating: 0)
        var top: Float = 0
        for i in 0..<3 {
            let axis = bins[i]
            // The busiest bin (the lower of equals, so the answer does not depend on dictionary order).
            guard let busiest = axis.max(by: { ($0.value, -$0.key) < ($1.value, -$1.key) }) else { return nil }
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
            busy[i] = (Float(busiest.key) + 0.5) * v
            if i == 1 {
                // The top layer: the highest level holding a quarter of the busiest level's hits.
                var level = last
                while level > busiest.key, (axis[level] ?? 0) * 4 < busiest.value { level -= 1 }
                top = Float(level + 1) * v
            }
        }
        return Extent(lo: lo, hi: hi, busiest: busy, top: top)
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
        takeDoors(yaw: yaw)
        var i = 0
        while i < tracks.count {
            var j = i + 1
            var mergedAny = false
            while j < tracks.count {
                let a = tracks[i], b = tracks[j]
                if !a.doubtful, !b.doubtful, spec.kinGroup(a.classIndex) == spec.kinGroup(b.classIndex), let ba = a.measured, let bb = b.measured,
                   Self.heightsOverlap(ba, bb),
                   Self.thickOverlap(ba, bb) >= spec.tracker.mergeOverlap || Self.adjoin(a, b, within: spec.tracker.adjoinMetres) {
                    // The older keeps its identity.
                    let (keep, drop) = a.born <= b.born ? (i, j) : (j, i)
                    trace?("frame \(frame): \(describe(tracks[keep])) merges \(describe(tracks[drop]))")
                    var survivor = tracks[keep]
                    let gone = tracks[drop]
                    for (k, n) in gone.voxels { survivor.voxels[k, default: Hits(count: 0, first: n.first, last: n.last)].add(n) }
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

    /// A curtain, window or mirror (or a doubtful thing) in the plane of a standing
    /// unit's front and adjoining it is one of its doors: it becomes part of the unit.
    private func takeDoors(yaw: Float) {
        var i = 0
        while i < tracks.count {
            var j = 0
            while j < tracks.count {
                if j != i, isStandingUnit(tracks[i]), isPanel(tracks[j]), let unit = tracks[i].extent, let panel = tracks[j].extent,
                   Self.isDoor(panel, of: unit) {
                    let door = tracks[j]
                    trace?("frame \(frame): \(describe(tracks[i])) takes \(describe(door)) as a door")
                    // (Its sightings do not count as the unit's: how settled the unit is, and so how
                    // many hits it asks of a voxel, goes by the times it was seen as itself.)
                    // A door seen more than once is vouched for twice over: by its own sightings and by
                    // lying where a door of this unit would. It need not also have been seen at length.
                    let vouched = door.observations >= spec.tracker.confirmObservations
                    for (k, n) in door.voxels {
                        var hits = n
                        hits.vouched = hits.vouched || vouched
                        tracks[i].voxels[k, default: Hits(count: 0, first: n.first, last: n.last)].add(hits)
                    }
                    tracks[i].lastSeen = max(tracks[i].lastSeen, door.lastSeen)
                    tracks[i].mergedIDs.insert(door.id)
                    tracks[i].mergedIDs.formUnion(door.mergedIDs)
                    tracks.remove(at: j)
                    if j < i { i -= 1 }
                    measure(&tracks[i], yaw: yaw)
                } else {
                    j += 1
                }
            }
            i += 1
        }
    }

    /// A wardrobe, dresser or bookshelf standing on the floor, seen often enough to be placed.
    private func isStandingUnit(_ track: Track) -> Bool {
        guard !track.doubtful, track.observations >= spec.tracker.confirmObservations, let info = spec.info(track.classIndex) else { return false }
        return info.family == "storage" && info.onFloor == true
    }

    private func isPanel(_ track: Track) -> Bool {
        track.doubtful || (spec.info(track.classIndex).map { Self.panelFamilies.contains($0.family) } ?? false)
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

    /// Two parts of one bed or one wardrobe: the same family, seen in parts, touching or
    /// nearly; and for storage in one plane (the wardrobe on one wall and the cabinet on
    /// the next meet in the corner, and are two).
    private static func adjoin(_ ta: Track, _ tb: Track, within reach: Float) -> Bool {
        guard let a = ta.measured, let b = tb.measured, a.family == b.family, seenInParts.contains(a.family) else { return false }
        if a.family == "storage" {
            // The smaller of the two is the part.
            guard let ea = ta.extent, let eb = tb.extent else { return false }
            let aIsPart = a.size.x * a.size.z <= b.size.x * b.size.z
            guard coplanar(aIsPart ? ea : eb, aIsPart ? eb : ea) else { return false }
        }
        let grow = SIMD3(reach, 0, reach)
        return RoomMapBuilder.footprintOverlap(a.min - grow, a.max + grow, b.min, b.max) > 0
    }

    /// Whether a part lies in the plane of a unit's front: across the part's
    /// thin side, the two are busiest within a hand of each other. (The unit
    /// may look deep, from clutter in front of it; the part, a door or two, is flat.)
    static func coplanar(_ part: Extent, _ unit: Extent) -> Bool {
        let size = part.hi - part.lo
        let axis = size.x <= size.z ? 0 : 2
        return abs(part.busiest[axis] - unit.busiest[axis]) <= 0.25
    }

    private static func thickOverlap(_ a: ObjectBox, _ b: ObjectBox) -> Float {
        let (aMin, aMax) = thickened(a.min, a.max), (bMin, bMax) = thickened(b.min, b.max)
        return RoomMapBuilder.footprintOverlap(aMin, aMax, bMin, bMax)
    }

    /// For diagnosis: along `axis` (0 x, 1 y, 2 z), per 5 cm bin, how many of an object's voxels
    /// are trusted (count as part of it) and how many are not, with their hit counts.
    public func hitProfile(of id: String, axis: Int) -> String {
        guard let track = tracks.first(where: { $0.id == id || $0.mergedIDs.contains(id) }) else { return "no \(id)" }
        let v = spec.tracker.voxelMetres, t = spec.tracker
        let established = track.observations >= 3
        var bins: [Int: (trusted: Int, all: Int, hits: Int)] = [:]
        for (k, h) in track.voxels {
            let key = axis == 0 ? Int(k.x) : axis == 1 ? Int(k.y) : Int(k.z)
            var b = bins[key] ?? (0, 0, 0)
            b.all += 1; b.hits += h.count
            let trusted = h.count >= 2 && (!established || h.vouched || h.last - h.first >= t.glimpseAnalyses || h.count >= 4 || 2 * h.count >= track.observations)
            if trusted { b.trusted += 1 }
            bins[key] = b
        }
        return "\(track.id) (\(track.observations) seen): " + bins.keys.sorted().map { k in
            String(format: "%+.2f:%d/%d(%d)", Float(k) * v, bins[k]!.trusted, bins[k]!.all, bins[k]!.hits)
        }.joined(separator: " ")
    }

    private func describe(_ track: Track) -> String {
        let label = spec.info(track.classIndex)?.label ?? "?"
        guard let b = track.measured else { return "\(track.id) \(label) (unmeasured)" }
        return String(format: "%@ %@ x %+.2f..%+.2f y %.2f..%.2f z %+.2f..%+.2f (%d seen)", track.id, label,
                      b.min.x, b.max.x, b.min.y, b.max.y, b.min.z, b.max.z, track.observations)
    }

    private func match(_ track: Track) -> ObservationMatch {
        let info = spec.info(track.classIndex)
        return ObservationMatch(objectID: track.id, label: info?.label ?? "", group: info?.group ?? "",
                                classIndex: track.classIndex,
                                placed: track.observations >= spec.tracker.confirmObservations && info?.boxed == true,
                                sightings: track.observations)
    }
}
