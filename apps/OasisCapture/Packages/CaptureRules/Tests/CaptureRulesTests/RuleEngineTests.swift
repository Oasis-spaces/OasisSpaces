import XCTest
import simd
@testable import CaptureRules

/// Simulated camera paths at 30 frames a second.
private let fps = 30.0

private func look(heading degrees: Double, pitch: Double = -5) -> SIMD3<Float> {
    let h = degrees * .pi / 180, p = pitch * .pi / 180
    return SIMD3<Float>(Float(sin(h) * cos(p)), Float(sin(p)), Float(-cos(h) * cos(p)))
}

private func frames(seconds: Double, start: Double = 0,
                    _ make: (Double) -> FrameSample) -> [FrameSample] {
    (0..<Int(seconds * fps)).map { make(start + Double($0) / fps) }
}

private func sample(_ t: Double, at position: SIMD3<Float> = .zero, heading: Double = 0,
                    pitch: Double = -5, tracking: FrameSample.Tracking = .normal,
                    exposure: Double? = 1.0 / 60, ambient: Double? = 1000, luma: Double? = 0.45,
                    saturated: Double? = 0.01, features: Int? = 200, nearest: Double? = 2.0,
                    people: Int? = 0) -> FrameSample {
    FrameSample(time: t, position: position, forward: look(heading: heading, pitch: pitch), tracking: tracking,
                exposureDuration: exposure, focalPixels: 1500, ambientIntensity: ambient, meanLuma: luma,
                saturatedShare: saturated, featurePoints: features, nearestSurface: nearest, peopleInView: people)
}

/// Runs samples through the engine; returns every rule shown at any point.
private func shown(_ engine: inout RuleEngine, _ samples: [FrameSample]) -> Set<Rule> {
    var rules = Set<Rule>()
    for s in samples {
        if let g = engine.update(s) { rules.insert(g.rule) }
    }
    return rules
}

final class RuleEngineTests: XCTestCase {
    func testBundledConfigLoadsWithTips() {
        let config = RuleConfig.bundled()
        XCTAssertEqual(config.coverageSectors, 12)
        XCTAssertFalse(config.tips.good.isEmpty)
        XCTAssertFalse(config.tips.best.isEmpty)
        for rule in Rule.allCases {
            XCTAssertNotNil(config.messages[rule.rawValue], "no message for \(rule)")
        }
    }

    func testCalmWalkShowsNothing() {
        var engine = RuleEngine()
        engine.startRecording()
        // Walk 0.3 m/s along a square while turning 20 degrees a second, looking down now and then.
        let walk = frames(seconds: 12) { t in
            sample(t, at: SIMD3<Float>(Float(0.3 * t), 0, 0), heading: 20 * t,
                   pitch: -5 - 30 * max(0, sin(t * .pi / 3)))
        }
        let rules = shown(&engine, walk)
        XCTAssertTrue(rules.isEmpty, "a calm walk should not be nagged: \(rules)")
    }

    func testPanFromOneSpotIsSpinning() {
        var engine = RuleEngine()
        engine.startRecording()
        // Standing still, turning 40 degrees a second: the pan video.
        let pan = frames(seconds: 8) { t in sample(t, heading: 40 * t, exposure: 1.0 / 120) }
        XCTAssertTrue(shown(&engine, pan).contains(.spinning))
    }

    func testWalkingWhileTurningIsNotSpinning() {
        var engine = RuleEngine()
        engine.startRecording()
        let walk = frames(seconds: 8) { t in
            sample(t, at: SIMD3<Float>(Float(0.4 * t), 0, 0), heading: 40 * t)
        }
        XCTAssertFalse(shown(&engine, walk).contains(.spinning))
    }

    func testFastTurnWarns() {
        var engine = RuleEngine()
        let whip = frames(seconds: 2) { t in sample(t, heading: 150 * t) }
        XCTAssertTrue(shown(&engine, whip).contains(.tooFast))
    }

    func testSlowTurnBlursInADarkRoom() {
        var engine = RuleEngine()
        // 30 degrees a second swings the view 3 degrees in a 1/10 s exposure, 0.25 in 1/120 s.
        let dim = frames(seconds: 2) { t in sample(t, heading: 30 * t, exposure: 0.1) }
        let bright = frames(seconds: 2, start: 10) { t in sample(t, heading: 30 * t, exposure: 1.0 / 120) }
        XCTAssertTrue(shown(&engine, dim).contains(.blurRisk))
        var engine2 = RuleEngine()
        XCTAssertFalse(shown(&engine2, bright).contains(.blurRisk))
    }

    func testBriefBlipIsDebounced() {
        var engine = RuleEngine()
        // Dark for 0.2 s only: shorter than holdSeconds, so nothing shows.
        let blip = frames(seconds: 2) { t in sample(t, ambient: t < 0.2 ? 50 : 1000) }
        XCTAssertFalse(shown(&engine, blip).contains(.tooDark))
    }

    func testDarkGlareCloseLowTextureAndPerson() {
        var e1 = RuleEngine()
        XCTAssertTrue(shown(&e1, frames(seconds: 2) { sample($0, ambient: 80, luma: 0.05) }).contains(.tooDark))
        var e2 = RuleEngine()
        XCTAssertTrue(shown(&e2, frames(seconds: 2) { sample($0, saturated: 0.4) }).contains(.glare))
        var e3 = RuleEngine()
        XCTAssertTrue(shown(&e3, frames(seconds: 2) { sample($0, nearest: 0.2) }).contains(.tooClose))
        var e4 = RuleEngine()
        XCTAssertTrue(shown(&e4, frames(seconds: 2) { sample($0, features: 5) }).contains(.lowTexture))
        var e5 = RuleEngine()
        // Detection runs every 15th frame; the answer carries over between runs.
        let people = frames(seconds: 2) { t in
            sample(t, people: Int((t * fps).rounded()) % 15 == 0 ? 1 : nil)
        }
        XCTAssertTrue(shown(&e5, people).contains(.person))
    }

    func testStopBeatsWarnAndShowsAtOnce() {
        var engine = RuleEngine()
        _ = shown(&engine, frames(seconds: 1) { sample($0, ambient: 80) })
        let g = engine.update(sample(1.0, tracking: .notAvailable, ambient: 80))
        XCTAssertEqual(g?.rule, .trackingLost)
        XCTAssertEqual(g?.isNew, true)
    }

    func testNeverLookingDownAsksForTheFloor() {
        var engine = RuleEngine()
        engine.startRecording()
        let level = frames(seconds: 12) { t in
            sample(t, at: SIMD3<Float>(Float(0.3 * t), 0, 0), heading: 20 * t, pitch: 0)
        }
        XCTAssertTrue(shown(&engine, level).contains(.lookDown))
    }

    func testFinishReportsCoverageAndAdvice() {
        var engine = RuleEngine()
        engine.startRecording()
        // A 30 s loop: full turn, looking down part of the time, ending at the start.
        let loop = frames(seconds: 30) { t in
            let a = t / 30 * 2 * .pi
            return sample(t, at: SIMD3<Float>(Float(1.5 * sin(a)), 0, Float(1.5 - 1.5 * cos(a))),
                          heading: 12 * t, pitch: -5 - 30 * max(0, sin(t * .pi / 2.5)))
        }
        _ = shown(&engine, loop)
        let (summary, advice) = engine.finish()
        XCTAssertEqual(summary.coverage, 1.0, accuracy: 0.01)
        XCTAssertTrue(summary.floorSeen)
        XCTAssertGreaterThan(summary.pathLength, 8)
        XCTAssertLessThan(summary.endDistanceFromStart, 0.5)
        XCTAssertTrue(advice.isEmpty, "a good loop needs no advice: \(advice)")

        var short = RuleEngine()
        short.startRecording()
        _ = shown(&short, frames(seconds: 6) { sample($0, heading: 10 * $0, pitch: 0) })
        let (_, shortAdvice) = short.finish()
        XCTAssertTrue(shortAdvice.contains(.tooShort))
        XCTAssertTrue(shortAdvice.contains(.lookDown))
        XCTAssertTrue(shortAdvice.contains(.coverage))
    }
}
