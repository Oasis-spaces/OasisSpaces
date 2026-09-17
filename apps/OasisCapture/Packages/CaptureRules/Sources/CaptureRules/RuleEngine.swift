import Foundation
import simd

/// A guidance rule. The raw value names its message in capture-rules.json.
public enum Rule: String, CaseIterable, Codable, Sendable {
    // Within a severity, earlier cases win: spinning in place ruins a scan
    // more than a moment of blur does.
    case trackingLost, relocalizing, initializing
    case spinning, tooFast, blurRisk
    case tooDark, glare, tooClose, lowTexture, person
    case lookDown, coverage
    case tooShort, closeLoop

    /// Most urgent first: only one message is shown at a time.
    public var severity: Severity {
        switch self {
        case .trackingLost, .relocalizing: return .stop
        case .tooFast, .blurRisk, .spinning, .tooDark, .glare, .tooClose, .lowTexture, .person: return .warn
        case .initializing, .lookDown, .coverage, .tooShort, .closeLoop: return .hint
        }
    }
}

public enum Severity: Int, Comparable, Codable, Sendable {
    case hint = 0, warn = 1, stop = 2
    public static func < (a: Severity, b: Severity) -> Bool { a.rawValue < b.rawValue }
}

/// The message on screen now.
public struct Guidance: Equatable, Sendable {
    public var rule: Rule
    public var message: String
    public var severity: Severity { rule.severity }
    /// True on the frame it first appears: the moment for a haptic.
    public var isNew: Bool
}

/// One stretch of time a rule held, for the capture report.
public struct RuleEpisode: Codable, Sendable {
    public var rule: Rule
    public var start: Double
    public var end: Double
}

public struct CaptureSummary: Codable, Sendable {
    public var duration: Double
    public var pathLength: Double
    /// Share of compass directions filmed at a usable pitch.
    public var coverage: Double
    public var floorSeen: Bool
    public var endDistanceFromStart: Double
    public var episodes: [RuleEpisode]
    /// Seconds each rule held in total.
    public var secondsByRule: [String: Double]
}

/// Turns the stream of FrameSamples into one guidance message at a time.
/// Conditions must hold for holdSeconds before a message shows and stop for
/// releaseSeconds before it clears, so the banner does not flicker.
public struct RuleEngine {
    public let config: RuleConfig
    public private(set) var current: Guidance?

    private var samples: [FrameSample] = []          // the recent window
    private var firstTime: Double?
    private var startPosition: SIMD3<Float>?
    private var lastPosition: SIMD3<Float>?
    private var pathLength: Double = 0
    private var minPitch: Double = 90
    private var sectorsSeen: Set<Int> = []
    private var smoothedAngular: Double = 0
    private var smoothedLinear: Double = 0
    private var holdingSince: [Rule: Double] = [:]
    private var lastHeld: [Rule: Double] = [:]
    private var visible: Set<Rule> = []
    private var episodes: [RuleEpisode] = []
    private var openEpisodes: [Rule: Double] = [:]
    private var recording = false

    public init(config: RuleConfig = .bundled()) {
        self.config = config
    }

    /// Start counting duration, path, coverage and episodes from here.
    public mutating func startRecording() {
        recording = true
        firstTime = nil
        startPosition = nil
        lastPosition = nil
        pathLength = 0
        minPitch = 90
        sectorsSeen = []
        episodes = []
        openEpisodes = [:]
    }

    public mutating func update(_ s: FrameSample) -> Guidance? {
        let previous = samples.last
        samples.append(s)
        samples.removeAll { s.time - $0.time > max(config.spinWindowSeconds, 2) }

        if let p = previous, s.time > p.time {
            let dt = s.time - p.time
            let turn = angleBetween(p.forward, s.forward)
            let travel = Double(simd_distance(p.position, s.position))
            smoothedAngular = ema(smoothedAngular, turn / dt, dt, tau: 0.25)
            smoothedLinear = ema(smoothedLinear, travel / dt, dt, tau: 0.25)
        }

        if recording {
            if firstTime == nil { firstTime = s.time; startPosition = s.position }
            if s.tracking == .normal {
                if let last = lastPosition { pathLength += Double(simd_distance(last, s.position)) }
                lastPosition = s.position
                minPitch = min(minPitch, s.pitch)
                if s.pitch >= config.coverageMinPitch && s.pitch <= config.coverageMaxPitch {
                    sectorsSeen.insert(Int(s.heading / 360 * Double(config.coverageSectors)) % config.coverageSectors)
                }
            }
        }

        let holding = conditions(s)
        return settle(holding, at: s.time)
    }

    /// Rules whose condition holds on this frame.
    private func conditions(_ s: FrameSample) -> Set<Rule> {
        var on = Set<Rule>()
        switch s.tracking {
        case .notAvailable: on.insert(.trackingLost)
        case .relocalizing: on.insert(.relocalizing)
        case .initializing: on.insert(.initializing)
        case .excessiveMotion: on.insert(.tooFast)
        case .insufficientFeatures: on.insert(.lowTexture)
        case .normal: break
        }
        if smoothedAngular > config.maxAngularSpeed || smoothedLinear > config.maxLinearSpeed {
            on.insert(.tooFast)
        }
        if let exposure = s.exposureDuration {
            // How far the view swings during one exposure. In degrees rather
            // than pixels, so the same threshold means the same smear on any
            // phone whatever its resolution.
            if smoothedAngular * exposure > config.maxBlurDegrees { on.insert(.blurRisk) }
        }
        if isSpinning(at: s.time) { on.insert(.spinning) }
        if let a = s.ambientIntensity, a < config.minAmbientIntensity { on.insert(.tooDark) }
        else if let l = s.meanLuma, l < config.minMeanLuma { on.insert(.tooDark) }
        if let sat = s.saturatedShare, sat > config.maxSaturatedShare { on.insert(.glare) }
        if let d = s.nearestSurface, d < config.minSurfaceDistance { on.insert(.tooClose) }
        if let n = s.featurePoints, n < config.minFeaturePoints, s.tracking == .normal { on.insert(.lowTexture) }
        if let people = s.peopleInView {
            if people > 0 { on.insert(.person) }
        } else if lastHeld[.person].map({ s.time - $0 < config.releaseSeconds }) == true {
            on.insert(.person)   // detection runs every few frames: keep the last answer
        }
        if recording, let t0 = firstTime {
            let elapsed = s.time - t0
            if elapsed > config.floorCheckAfterSeconds && minPitch > config.floorPitch { on.insert(.lookDown) }
            if elapsed > config.coverageHintAfterSeconds && coverage < config.coverageGoal { on.insert(.coverage) }
        }
        return on
    }

    /// Turned a lot while hardly moving: a pan from one spot, which
    /// reconstructs poorly (every view shares one centre).
    private func isSpinning(at time: Double) -> Bool {
        guard let first = samples.first, time - first.time >= config.spinWindowSeconds * 0.9 else { return false }
        var turn = 0.0
        var travel = 0.0
        for (a, b) in zip(samples, samples.dropFirst()) {
            turn += headingChange(a, b)
            travel = max(travel, Double(simd_distance(first.position, b.position)))
        }
        return turn > config.spinMinTurn && travel < config.spinMaxTravel
    }

    /// Debounce: a rule shows once its condition has held for holdSeconds
    /// (stop rules at once), and stays until the condition has been off for
    /// releaseSeconds. The most severe showing rule wins.
    private mutating func settle(_ holding: Set<Rule>, at time: Double) -> Guidance? {
        for rule in Rule.allCases {
            if holding.contains(rule) {
                if holdingSince[rule] == nil { holdingSince[rule] = time }
                lastHeld[rule] = time
            } else {
                holdingSince[rule] = nil
            }
        }
        var showing = Set<Rule>()
        for rule in Rule.allCases {
            if let start = holdingSince[rule],
               time - start >= (rule.severity == .stop ? 0 : config.holdSeconds) {
                showing.insert(rule)
            } else if visible.contains(rule), let last = lastHeld[rule], time - last <= config.releaseSeconds {
                showing.insert(rule)   // just stopped: linger so the banner does not flicker
            }
        }
        visible = showing
        if recording {
            for rule in showing where openEpisodes[rule] == nil { openEpisodes[rule] = time }
            for (rule, start) in openEpisodes where !showing.contains(rule) {
                episodes.append(RuleEpisode(rule: rule, start: start - (firstTime ?? start), end: time - (firstTime ?? time)))
                openEpisodes[rule] = nil
            }
        }
        guard let top = showing.max(by: { a, b in
            a.severity != b.severity ? a.severity < b.severity
                : Rule.allCases.firstIndex(of: a)! > Rule.allCases.firstIndex(of: b)!
        }) else {
            current = nil
            return nil
        }
        let guidance = Guidance(rule: top, message: config.message(top), isNew: current?.rule != top)
        current = guidance
        return guidance
    }

    public var coverage: Double {
        Double(sectorsSeen.count) / Double(config.coverageSectors)
    }

    public var sectors: Set<Int> { sectorsSeen }

    /// The camera has looked down far enough to see the floor meet the walls.
    public var floorSeen: Bool { minPitch <= config.floorPitch }

    public var elapsed: Double {
        guard let t0 = firstTime, let last = samples.last else { return 0 }
        return last.time - t0
    }

    /// The report of a finished recording, with the end-of-recording checks.
    public mutating func finish() -> (summary: CaptureSummary, advice: [Rule]) {
        let end = samples.last?.time ?? 0
        for (rule, start) in openEpisodes {
            episodes.append(RuleEpisode(rule: rule, start: start - (firstTime ?? start), end: end - (firstTime ?? end)))
        }
        openEpisodes = [:]
        recording = false
        let duration = elapsed
        let endDistance = Double(simd_distance(startPosition ?? .zero, samples.last?.position ?? .zero))
        var seconds: [String: Double] = [:]
        for e in episodes { seconds[e.rule.rawValue, default: 0] += e.end - e.start }
        var advice: [Rule] = []
        if duration < config.minDurationSeconds { advice.append(.tooShort) }
        if minPitch > config.floorPitch { advice.append(.lookDown) }
        if coverage < config.coverageGoal { advice.append(.coverage) }
        if endDistance > config.loopCloseDistance { advice.append(.closeLoop) }
        let summary = CaptureSummary(duration: duration, pathLength: pathLength, coverage: coverage,
                                     floorSeen: minPitch <= config.floorPitch,
                                     endDistanceFromStart: endDistance,
                                     episodes: episodes.sorted { $0.start < $1.start },
                                     secondsByRule: seconds)
        return (summary, advice)
    }
}

private func angleBetween(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Double {
    let d = Double(simd_dot(simd_normalize(a), simd_normalize(b)))
    return acos(max(-1, min(1, d))) * 180 / .pi
}

private func headingChange(_ a: FrameSample, _ b: FrameSample) -> Double {
    var d = abs(b.heading - a.heading)
    if d > 180 { d = 360 - d }
    return d
}

private func ema(_ previous: Double, _ value: Double, _ dt: Double, tau: Double) -> Double {
    let k = min(1, dt / tau)
    return previous + (value - previous) * k
}
