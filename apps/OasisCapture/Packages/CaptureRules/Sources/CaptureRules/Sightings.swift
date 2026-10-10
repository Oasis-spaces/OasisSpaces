import Foundation
import simd

/// One detected thing in one frame, as the direction it was seen in.
public struct Sighting: Sendable {
    public var classIndex: Int
    public var confidence: Float
    /// Unit vector from the camera to the thing's centre, in the room.
    public var direction: SIMD3<Float>
    /// Half the thing's apparent size, radians.
    public var radius: Float

    public init(classIndex: Int, confidence: Float, direction: SIMD3<Float>, radius: Float) {
        self.classIndex = classIndex
        self.confidence = confidence
        self.direction = direction
        self.radius = radius
    }

    /// From a thing's centre and size in upright normalised image coordinates
    /// (x right, y down). `sensorLandscape`: the camera's image is the phone's
    /// landscape sensor image, which shows upright (x, y) at sensor (y, 1 - x).
    public init(classIndex: Int, confidence: Float, centre: SIMD2<Float>, size: SIMD2<Float>, camera: PinholeCamera,
                sensorLandscape: Bool) {
        let w = Float(camera.width), h = Float(camera.height)
        let u = sensorLandscape ? centre.y * w : centre.x * w
        let v = sensorLandscape ? (1 - centre.x) * h : centre.y * h
        let pixels = sensorLandscape ? SIMD2(size.y * w, size.x * h) : SIMD2(size.x * w, size.y * h)
        self.init(classIndex: classIndex, confidence: confidence, direction: camera.direction(u: u, v: v),
                  radius: atan(simd_length(pixels) / 2 / camera.fx))
    }
}

/// Remembers what was detected in the last few analysed frames by the
/// direction it was seen in, for two things the screen needs before the room
/// map knows an object: a thing is only outlined once it has been seen twice
/// (the detector's one-frame inventions never show), and its name is the one
/// most voted so far (a wardrobe one frame called a shelf stays a wardrobe).
///
/// Direction, not place in the image: between two analyses the picture slides
/// by more than a small thing's width, but the camera's turn is known, so the
/// same thing is found in the same direction, give or take the step taken.
/// It needs no depth, so it works on a bare wall too.
public struct SightingMemory {
    public struct Verdict: Sendable, Equatable {
        /// Times the thing has been seen in this direction lately, this time included.
        public var sightings: Int
        /// The class most voted for it.
        public var classIndex: Int
    }

    private struct Chain {
        var kin: String
        var direction: SIMD3<Float>
        var radius: Float
        var eye: SIMD3<Float>
        var lastSeen: Int
        var count: Int
        var votes: [Int: Float]
        var classIndex: Int
    }

    public let spec: ObjectSpec
    private var chains: [Chain] = []
    private var analysis = 0
    /// Told of every chain a sighting joins, with the votes so far (the simulator's --trace).
    public var trace: ((String) -> Void)?

    public init(spec: ObjectSpec) {
        self.spec = spec
    }

    public mutating func reset() {
        chains = []
        analysis = 0
    }

    /// One analysed frame's sightings, taken from `eye`. Returns a verdict for each, in order.
    public mutating func observe(_ sightings: [Sighting], eye: SIMD3<Float>) -> [Verdict] {
        analysis += 1
        let screen = spec.screen
        chains.removeAll { analysis - $0.lastSeen > screen.forgetAnalyses }

        // Pair each sighting with the remembered thing nearest in direction, closest first: one of
        // its kin anywhere within the thing's size, or one of any kind seen as the same outline
        // (a TV one frame called a window, a wardrobe called a door: one place, one name).
        struct Pair { var sighting: Int; var chain: Int; var score: Float }
        var pairs: [Pair] = []
        for (si, s) in sightings.enumerated() {
            guard let kin = spec.kinGroup(s.classIndex) else { continue }
            for (ci, c) in chains.enumerated() {
                let angle = acos(max(-1, min(1, simd_dot(s.direction, c.direction))))
                // As far apart as half the thing's size, plus what a step sideways moves something a metre away.
                let step = atan(simd_distance(eye, c.eye))
                let allowed = max(screen.minAngleDegrees * .pi / 180, 0.75 * max(s.radius, c.radius)) + step
                if c.kin == kin {
                    if angle <= allowed { pairs.append(Pair(sighting: si, chain: ci, score: angle / allowed)) }
                } else if Self.sameOutline(angle: angle, step: step, s.radius, c.radius) {
                    pairs.append(Pair(sighting: si, chain: ci, score: angle / allowed))
                }
            }
        }
        pairs.sort { $0.score < $1.score }
        var chainOf = [Int?](repeating: nil, count: sightings.count)
        var taken = Set<Int>()
        for pair in pairs where chainOf[pair.sighting] == nil && !taken.contains(pair.chain) {
            chainOf[pair.sighting] = pair.chain
            taken.insert(pair.chain)
        }

        var verdicts: [Verdict] = []
        for (si, s) in sightings.enumerated() {
            guard let kin = spec.kinGroup(s.classIndex) else {
                verdicts.append(Verdict(sightings: 1, classIndex: s.classIndex))
                continue
            }
            if let ci = chainOf[si] {
                chains[ci].direction = s.direction
                chains[ci].radius = s.radius
                chains[ci].eye = eye
                chains[ci].lastSeen = analysis
                chains[ci].count += 1
                // The latest looks weigh most: a TV called a painting from across the room is a TV up close.
                for key in chains[ci].votes.keys { chains[ci].votes[key]! *= screen.voteDecay }
                chains[ci].votes[s.classIndex, default: 0] += s.confidence
                // The most voted name; the present one keeps a tie. The name's kin is the chain's.
                let current = chains[ci].votes[chains[ci].classIndex] ?? 0
                if let best = chains[ci].votes.max(by: { ($0.value, -$0.key) < ($1.value, -$1.key) }), best.value > current {
                    chains[ci].classIndex = best.key
                    chains[ci].kin = spec.kinGroup(best.key) ?? chains[ci].kin
                }
                if let trace {
                    let votes = chains[ci].votes.sorted { $0.value > $1.value }.map { String(format: "%@ %.2f", spec.info($0.key)?.label ?? "?", $0.value) }
                    trace("analysis \(analysis): \(spec.info(s.classIndex)?.label ?? "?") joins chain \(ci) as \(spec.info(chains[ci].classIndex)?.label ?? "?") (\(chains[ci].count) sightings; \(votes.joined(separator: ", ")))")
                }
                verdicts.append(Verdict(sightings: chains[ci].count, classIndex: chains[ci].classIndex))
            } else {
                chains.append(Chain(kin: kin, direction: s.direction, radius: s.radius, eye: eye, lastSeen: analysis, count: 1,
                                    votes: [s.classIndex: s.confidence], classIndex: s.classIndex))
                verdicts.append(Verdict(sightings: 1, classIndex: s.classIndex))
            }
        }
        return verdicts
    }

    /// Two sightings of different kinds are one thing when they are the same
    /// outline: centres within a third of the thing's apparent size (plus the
    /// step taken) and sizes alike. A pillow inside a bed's outline is not.
    static func sameOutline(angle: Float, step: Float, _ a: Float, _ b: Float) -> Bool {
        let (small, big) = (min(a, b), max(a, b))
        return small > 0 && big <= 1.4 * small && angle <= 0.35 * big + step
    }
}

extension ObjectSpec {
    /// Whether a thing's outline goes on the screen: seen often enough, in
    /// this direction lately (`sightings`) or as an object of the room map
    /// (`tracked`: coming back to a known piece shows it at once).
    public func shows(sightings: Int, tracked: Int) -> Bool {
        max(sightings, tracked) >= screen.showSightings
    }
}
