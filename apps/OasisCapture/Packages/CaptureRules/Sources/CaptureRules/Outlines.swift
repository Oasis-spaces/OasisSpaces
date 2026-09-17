import Foundation
import simd

/// The segmentation model's classes: which are outlined, their group and colour.
public struct DetectionSpec: Codable, Sendable {
    public struct Group: Codable, Sendable {
        public var color: String
    }

    public struct ClassInfo: Codable, Sendable {
        public var id: Int
        public var name: String
        public var label: String
        public var group: String
        public var outline: Bool
        /// Look-alikes the model flips between share a family (sofa and armchair).
        public var family: String?
        public var familyName: String { family ?? name }
    }

    public var model: String
    public var groups: [String: Group]
    /// A region smaller than this share of the image is not outlined.
    public var minRegionShare: Double
    /// A person covering this share of the image triggers the warning.
    public var personWarnShare: Double
    public var classes: [ClassInfo]
    // Room map tuning (see RoomMapBuilder).
    public var voteVoxelMetres: Float = 0.08
    public var groupCellMetres: Float = 0.25
    public var minVoxels: Int = 12
    public var confirmBuilds: Int = 2
    public var graceBuilds: Int = 3
    public var labelHistoryFrames: Int = 6
    public var maxObjectMetres: Float = 4.0
    public var pointDepthMetres: [Float] = [0.3, 6.0]

    public static func bundled() -> DetectionSpec {
        guard let url = Bundle.module.url(forResource: "detection-classes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let spec = try? JSONDecoder().decode(DetectionSpec.self, from: data) else {
            fatalError("detection-classes.json is missing or does not match DetectionSpec")
        }
        return spec
    }

    public func info(_ id: Int) -> ClassInfo? {
        id >= 0 && id < classes.count && classes[id].id == id ? classes[id] : classes.first { $0.id == id }
    }

    /// The family a class belongs to (its own name when it has none).
    public func family(_ id: Int) -> String? {
        info(id)?.familyName
    }
}

/// Remembers what recent frames called each place on screen, so a label does
/// not flip between look-alikes from one frame to the next: a region takes the
/// class most seen at its centre over the last few frames.
public struct LabelMemory {
    private var maps: [(classes: [Int32], width: Int, height: Int)] = []
    private let depth: Int
    private let spec: DetectionSpec

    public init(spec: DetectionSpec) {
        self.spec = spec
        depth = max(1, spec.labelHistoryFrames)
    }

    public mutating func reset() { maps.removeAll() }

    /// Adds the newest result and returns it with steadied labels (upright coordinates).
    public mutating func steady(_ result: SegmentationResult) -> SegmentationResult {
        maps.append((result.classes, result.width, result.height))
        if maps.count > depth { maps.removeFirst() }
        guard maps.count > 1 else { return result }
        var steadied = result
        for i in steadied.regions.indices {
            let c = steadied.regions[i].centroid
            var votes: [Int: Int] = [:]
            for map in maps {
                let x = Int(c.x * Double(map.width)), y = Int(c.y * Double(map.height))
                guard x >= 0, y >= 0, x < map.width, y < map.height else { continue }
                votes[Int(map.classes[y * map.width + x]), default: 0] += 1
            }
            votes[steadied.regions[i].classId, default: 0] += 1   // the frame itself counts too
            // Only classes of the same family may replace the frame's own answer:
            // a chair flips to sofa, not to floor.
            let family = spec.family(steadied.regions[i].classId)
            let best = votes.filter { spec.family($0.key) == family }.max { $0.value < $1.value }
            if let best, let info = spec.info(best.key), best.key != steadied.regions[i].classId {
                steadied.regions[i].classId = best.key
                steadied.regions[i].label = info.label
                steadied.regions[i].group = info.group
            }
        }
        return steadied
    }
}

/// One outlined thing in a frame.
public struct Region: Sendable {
    public var classId: Int
    public var label: String
    public var group: String
    /// Share of the image it covers.
    public var share: Double
    /// Normalised image coordinates, x right and y down, 0...1.
    public var centroid: SIMD2<Double>
    /// Closed polygon around it, same coordinates.
    public var outline: [SIMD2<Double>]
}

public struct SegmentationResult: Sendable {
    public var regions: [Region]
    /// Share of the image covered by people.
    public var personShare: Double
    /// Share of the image per class id, for every class present.
    public var classShares: [Int: Double]
    /// The class map itself (row-major, upright), to label 3D points with.
    public var classes: [Int32]
    public var width: Int
    public var height: Int

    /// The class at a point of the upright image, 0...1 coordinates.
    public func classAt(x: Double, y: Double) -> Int? {
        guard x >= 0, y >= 0, x < 1, y < 1 else { return nil }
        return Int(classes[Int(y * Double(height)) * width + Int(x * Double(width))])
    }
}

public enum OutlineExtractor {
    /// Outlines of every outlined class's connected regions in a class map
    /// (row-major, width x height), largest first.
    public static func extract(classes: [Int32], width: Int, height: Int, spec: DetectionSpec,
                               maxRegions: Int = 24, simplify epsilon: Double = 1.2) -> SegmentationResult {
        let total = width * height
        precondition(classes.count == total, "class map size does not match")
        var counts: [Int32: Int] = [:]
        for c in classes { counts[c, default: 0] += 1 }
        let shares = Dictionary(uniqueKeysWithValues: counts.map { (Int($0.key), Double($0.value) / Double(total)) })
        let personId = spec.classes.first { $0.group == "person" }?.id
        let personShare = personId.flatMap { shares[$0] } ?? 0

        let minArea = max(1, Int(spec.minRegionShare * Double(total)))
        var visited = [Bool](repeating: false, count: total)
        var regions: [Region] = []
        var queue: [Int] = []
        queue.reserveCapacity(total)

        for start in 0..<total where !visited[start] {
            let cls = classes[start]
            guard let info = spec.info(Int(cls)), info.outline, (counts[cls] ?? 0) >= minArea else {
                visited[start] = true
                continue
            }
            // Flood fill the connected region (4-neighbours).
            queue.removeAll(keepingCapacity: true)
            queue.append(start)
            visited[start] = true
            var head = 0
            var sumX = 0.0, sumY = 0.0
            while head < queue.count {
                let i = queue[head]
                head += 1
                let x = i % width, y = i / width
                sumX += Double(x)
                sumY += Double(y)
                for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                where nx >= 0 && ny >= 0 && nx < width && ny < height {
                    let j = ny * width + nx
                    if !visited[j] && classes[j] == cls {
                        visited[j] = true
                        queue.append(j)
                    }
                }
            }
            let area = queue.count
            guard area >= minArea else { continue }
            // The flood started at the region's first pixel in raster order,
            // which is where boundary tracing must start.
            let boundary = traceBoundary(start: start, width: width, height: height) { classes[$0] == cls }
            let simplified = simplifyClosed(boundary.map { SIMD2(Double($0 % width), Double($0 / width)) }, epsilon: epsilon)
            regions.append(Region(
                classId: Int(cls), label: info.label, group: info.group,
                share: Double(area) / Double(total),
                centroid: SIMD2((sumX / Double(area) + 0.5) / Double(width), (sumY / Double(area) + 0.5) / Double(height)),
                outline: simplified.map { SIMD2(($0.x + 0.5) / Double(width), ($0.y + 0.5) / Double(height)) }))
        }
        regions.sort { $0.share > $1.share }
        return SegmentationResult(regions: Array(regions.prefix(maxRegions)), personShare: personShare,
                                  classShares: shares, classes: classes, width: width, height: height)
    }

    /// Moore-neighbour tracing of a region's outer boundary, clockwise (y down).
    static func traceBoundary(start: Int, width: Int, height: Int, inside: (Int) -> Bool) -> [Int] {
        // Clockwise from west, with y pointing down.
        let dx = [-1, -1, 0, 1, 1, 1, 0, -1]
        let dy = [0, -1, -1, -1, 0, 1, 1, 1]
        func member(_ x: Int, _ y: Int) -> Bool {
            x >= 0 && y >= 0 && x < width && y < height && inside(y * width + x)
        }
        var boundary = [start]
        var px = start % width, py = start / width
        // Entered from the west: the pixel before start in raster order is outside.
        var back = 0
        let limit = 4 * width * height
        var steps = 0
        var secondPixel: Int?
        while steps < limit {
            steps += 1
            var found = false
            for k in 1...8 {
                let d = (back + k) % 8
                let nx = px + dx[d], ny = py + dy[d]
                if member(nx, ny) {
                    let next = ny * width + nx
                    // Jacob's stopping criterion: back at the start, about to
                    // repeat the first move.
                    if next == secondPixel && py * width + px == start { return boundary.dropLast().isEmpty ? boundary : Array(boundary.dropLast()) }
                    if secondPixel == nil { secondPixel = next }
                    // The neighbour checked just before this one is outside; look
                    // from the new pixel back towards it.
                    let prev = (d + 7) % 8
                    let bx = px + dx[prev], by = py + dy[prev]
                    px = nx
                    py = ny
                    back = direction(from: (px, py), to: (bx, by), dx: dx, dy: dy)
                    boundary.append(next)
                    found = true
                    break
                }
            }
            if !found { return boundary }   // a single isolated pixel
        }
        return boundary
    }

    private static func direction(from p: (Int, Int), to q: (Int, Int), dx: [Int], dy: [Int]) -> Int {
        let ddx = q.0 - p.0, ddy = q.1 - p.1
        for d in 0..<8 where dx[d] == ddx && dy[d] == ddy { return d }
        return 0
    }

    /// Douglas-Peucker on a closed polygon: split at the point farthest from
    /// the first, simplify both halves.
    static func simplifyClosed(_ points: [SIMD2<Double>], epsilon: Double) -> [SIMD2<Double>] {
        guard points.count > 4 else { return points }
        let first = points[0]
        let far = points.indices.max { simd_distance(points[$0], first) < simd_distance(points[$1], first) }!
        let a = simplifyOpen(Array(points[0...far]), epsilon: epsilon)
        let b = simplifyOpen(Array(points[far...]) + [first], epsilon: epsilon)
        return Array(a.dropLast()) + Array(b.dropLast())
    }

    private static func simplifyOpen(_ points: [SIMD2<Double>], epsilon: Double) -> [SIMD2<Double>] {
        guard points.count > 2 else { return points }
        let a = points.first!, b = points.last!
        var maxDistance = 0.0, index = 0
        for i in 1..<(points.count - 1) {
            let d = distanceToSegment(points[i], a, b)
            if d > maxDistance { maxDistance = d; index = i }
        }
        guard maxDistance > epsilon else { return [a, b] }
        let left = simplifyOpen(Array(points[0...index]), epsilon: epsilon)
        let right = simplifyOpen(Array(points[index...]), epsilon: epsilon)
        return Array(left.dropLast()) + right
    }

    private static func distanceToSegment(_ p: SIMD2<Double>, _ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
        let ab = b - a
        let length = simd_length_squared(ab)
        guard length > 0 else { return simd_distance(p, a) }
        let t = max(0, min(1, simd_dot(p - a, ab) / length))
        return simd_distance(p, a + t * ab)
    }
}
