import XCTest
import simd
@testable import CaptureRules

/// The name a thing gets, and keeps: the detector's one-frame confusions
/// (a chair's back as a toilet, a lamp as a person, a wardrobe as a door, a
/// TV as a window) are undone by what the thing is, not by which room it is in.
final class LabelTests: XCTestCase {
    let spec = ObjectSpec.bundled()

    private func cls(_ name: String) -> Int {
        spec.index(of: name)!
    }

    private func instance(_ name: String, minX: Float, minY: Float, maxX: Float, maxY: Float,
                          confidence: Float = 0.6, alternatives: [(String, Float)] = []) -> Instance {
        let w = 64, h = 64
        var mask = [UInt8](repeating: 0, count: w * h)
        var area = 0
        for y in Int(minY * Float(h))..<Int(maxY * Float(h)) {
            for x in Int(minX * Float(w))..<Int(maxX * Float(w)) { mask[y * w + x] = 1; area += 1 }
        }
        return Instance(classIndex: cls(name), confidence: confidence, minX: minX, minY: minY, maxX: maxX, maxY: maxY,
                        mask: mask, maskWidth: w, maskHeight: h, area: area,
                        alternatives: alternatives.map { Instance.Alternative(classIndex: cls($0.0), confidence: $0.1) })
    }

    /// An upright 4:3 camera: 1440 x 1920, focal 0.75 of the height.
    private let upright = PinholeCamera(fx: 1440, fy: 1440, cx: 720, cy: 960, width: 1440, height: 1920, transform: matrix_identity_float4x4)

    // MARK: The decoder keeps the detector's other names for an outline

    func testDecoderKeepsTheRunnerUpNameOfAnOutline() {
        let nc = spec.classes.count, channels = 4 + nc + 32, anchors = 2
        var predictions = [Float](repeating: 0, count: channels * anchors)
        func set(_ channel: Int, _ anchor: Int, _ value: Float) { predictions[channel * anchors + anchor] = value }
        // Two anchors on the same box: one says chair 0.5 with toilet 0.3 behind it, the other says toilet 0.45.
        for a in 0..<anchors {
            set(0, a, 320); set(1, a, 320); set(2, a, 200); set(3, a, 200)
            set(4 + nc, a, 8)                                            // mask coefficient 0: everything inside
        }
        set(4 + cls("chair"), 0, 0.5); set(4 + cls("toilet"), 0, 0.3)
        set(4 + cls("toilet"), 1, 0.45)
        let maskWidth = 16, maskHeight = 16
        let protos = [Float](repeating: 1, count: 32 * maskWidth * maskHeight)
        let instances = InstanceDecoder.decode(predictions: predictions, anchors: anchors, protos: protos,
                                          maskWidth: maskWidth, maskHeight: maskHeight, spec: spec)
        XCTAssertEqual(instances.count, 1, "one outline, not one per name")
        XCTAssertEqual(instances.first?.classIndex, cls("chair"))
        XCTAssertEqual(instances.first?.alternatives.map { $0.classIndex }, [cls("toilet")])
        XCTAssertEqual(instances.first?.alternatives.first?.confidence ?? 0, 0.45, accuracy: 0.001, "the strongest of its toilet scores")
    }

    // MARK: What a thing can be, by its outline and size

    func testAStripAlongTheFramesEdgeGetsNoNameYet() {
        // A chair's back showing as a strip along the bottom edge, which the detector called a toilet.
        let strip = instance("toilet", minX: 0.3, minY: 0.93, maxX: 0.57, maxY: 1.0)
        XCTAssertNil(spec.believed(strip, size: (0.5, 0.15)))
        // The same thing with a fair part of it in view keeps its name.
        let more = instance("chair", minX: 0.3, minY: 0.6, maxX: 0.57, maxY: 1.0)
        XCTAssertEqual(spec.believed(more, size: (0.5, 0.9)), cls("chair"))
    }

    func testAPersonTooSmallToBeOneIsTheDetectorsRunnerUp() {
        let lamp = instance("person", minX: 0.4, minY: 0.3, maxX: 0.5, maxY: 0.45, alternatives: [("lamp", 0.3)])
        XCTAssertEqual(spec.believed(lamp, size: (0.2, 0.4)), cls("lamp"))
        // With no other name above the threshold it gets none.
        let nameless = instance("person", minX: 0.4, minY: 0.3, maxX: 0.5, maxY: 0.45, alternatives: [("lamp", 0.1)])
        XCTAssertNil(spec.believed(nameless, size: (0.2, 0.4)))
        // A real person, or one of unknown size, is a person.
        XCTAssertEqual(spec.believed(lamp, size: (0.5, 1.7)), cls("person"))
        XCTAssertEqual(spec.believed(lamp, size: nil), cls("person"))
    }

    func testAMinimumIsNotHeldAgainstAThingCutByTheFrame() {
        // A wardrobe whose top is above the frame measures 0.6 m tall in view, and is still a wardrobe.
        let cut = instance("wardrobe", minX: 0.2, minY: 0.0, maxX: 0.6, maxY: 0.5)
        XCTAssertEqual(spec.believed(cut, size: (1.2, 0.6)), cls("wardrobe"))
        // Seen whole at 0.6 m, it is not a wardrobe; the detector's cabinet is taken instead.
        let whole = instance("wardrobe", minX: 0.2, minY: 0.2, maxX: 0.6, maxY: 0.5, alternatives: [("cabinet", 0.35)])
        XCTAssertEqual(spec.believed(whole, size: (1.2, 0.6)), cls("cabinet"))
    }

    func testApparentSizeComesFromTheDepthInsideTheOutline() {
        let thing = instance("lamp", minX: 0.4, minY: 0.3, maxX: 0.5, maxY: 0.45)
        let size = thing.apparentSize(camera: upright, sensorLandscape: false) { _, _ in 2.0 }
        XCTAssertNotNil(size)
        XCTAssertEqual(size?.width ?? 0, 0.1 * 1440 / 1440 * 2, accuracy: 0.01)
        XCTAssertEqual(size?.height ?? 0, 0.15 * 1920 / 1440 * 2, accuracy: 0.01)
        XCTAssertEqual(size?.depth ?? 0, 2, accuracy: 0.001)
    }

    // MARK: One place, one name

    private func sighting(_ name: String, confidence: Float, degrees: Float, radiusDegrees: Float) -> Sighting {
        let a = degrees * .pi / 180
        return Sighting(classIndex: cls(name), confidence: confidence, direction: SIMD3(sin(a), 0, -cos(a)), radius: radiusDegrees * .pi / 180)
    }

    func testTheSameOutlineUnderAnotherNameJoinsTheChainAndTheVotesName() {
        var memory = SightingMemory(spec: spec)
        // A TV seen twice, then the same outline called a window, then a mirror: still the TV, and shown.
        _ = memory.observe([sighting("TV", confidence: 0.6, degrees: 10, radiusDegrees: 8)], eye: .zero)
        _ = memory.observe([sighting("TV", confidence: 0.6, degrees: 11, radiusDegrees: 8)], eye: .zero)
        let asWindow = memory.observe([sighting("window", confidence: 0.5, degrees: 11, radiusDegrees: 9)], eye: .zero)
        XCTAssertEqual(asWindow.first?.classIndex, cls("TV"))
        XCTAssertEqual(asWindow.first?.sightings, 3)
        let asMirror = memory.observe([sighting("mirror", confidence: 0.6, degrees: 12, radiusDegrees: 8)], eye: .zero)
        XCTAssertEqual(asMirror.first?.classIndex, cls("TV"))
        // A wardrobe one frame called a door, the next frames a wardrobe: a wardrobe from the second sighting.
        _ = memory.observe([sighting("door", confidence: 0.5, degrees: 40, radiusDegrees: 12)], eye: .zero)
        let second = memory.observe([sighting("wardrobe", confidence: 0.6, degrees: 41, radiusDegrees: 12)], eye: .zero)
        XCTAssertEqual(second.first?.classIndex, cls("wardrobe"))
        XCTAssertEqual(second.first?.sightings, 2)
    }

    func testAPillowOnABedIsNotTheBed() {
        var memory = SightingMemory(spec: spec)
        _ = memory.observe([sighting("bed", confidence: 0.8, degrees: 0, radiusDegrees: 25)], eye: .zero)
        let both = memory.observe([sighting("bed", confidence: 0.8, degrees: 0, radiusDegrees: 25),
                                   sighting("pillow", confidence: 0.6, degrees: 2, radiusDegrees: 5)], eye: .zero)
        XCTAssertEqual(both[0].classIndex, cls("bed"))
        XCTAssertEqual(both[0].sightings, 2)
        XCTAssertEqual(both[1].classIndex, cls("pillow"))
        XCTAssertEqual(both[1].sightings, 1, "its own chain, not the bed's")
    }

    // MARK: Two tracks that are one box

    private func points(from lo: SIMD3<Float>, to hi: SIMD3<Float>) -> [SIMD3<Float>] {
        var out: [SIMD3<Float>] = []
        var x = lo.x
        while x <= hi.x + 1e-4 {
            var y = lo.y
            while y <= hi.y + 1e-4 {
                var z = lo.z
                while z <= hi.z + 1e-4 { out.append(SIMD3(x, y, z)); z += 0.04 }
                y += 0.04
            }
            x += 0.04
        }
        return out
    }

    private func observation(_ name: String, from lo: SIMD3<Float>, to hi: SIMD3<Float>, confidence: Float = 0.7) -> ObjectObservation {
        ObjectObservation(classIndex: cls(name), confidence: confidence, points: points(from: lo, to: hi))
    }

    private func seeFloor(_ builder: RoomMapBuilder) {
        builder.saw(floorAt: (0..<300).map { Float($0 % 3) * 0.01 }, pointsAt: (0..<3000).map { Float($0 % 250) * 0.01 })
    }

    func testTheSameBoxUnderTwoNamesIsOneObjectNamedByTheVotes() {
        let builder = RoomMapBuilder(spec: spec)
        seeFloor(builder)
        let lo = SIMD3<Float>(0, 0, 0), hi = SIMD3<Float>(1.0, 1.9, 0.5)
        // A wardrobe seen three times, and twice as a heater (another family) in the same place.
        for _ in 0..<3 { builder.observe([observation("wardrobe", from: lo, to: hi)], camera: nil) }
        for _ in 0..<2 { builder.observe([observation("heater", from: lo, to: hi, confidence: 0.5)], camera: nil) }
        let objects = builder.build().objects
        XCTAssertEqual(objects.count, 1, "\(objects.map(\.label))")
        XCTAssertEqual(objects.first?.label, "wardrobe")
    }

    func testAGlimpseAddsToWhatItLiesOnAndStartsNothing() {
        let builder = RoomMapBuilder(spec: spec)
        seeFloor(builder)
        // A bed seen three times by its near part; then, cut by the frame's edge, a stretch of it
        // reaching further, which the detector called a sofa and a bathtub: the bed grows to that
        // length, under its own name.
        for n in 0..<12 {                                          // the box eases towards what it sees
            if n % 2 == 0 {
                builder.observe([observation("bed", from: SIMD3(0, 0, 0), to: SIMD3(1.4, 0.5, 1.0))], camera: nil)
            } else {
                var far = observation(n % 4 == 1 ? "sofa" : "bathtub", from: SIMD3(0, 0, 0.5), to: SIMD3(1.4, 0.5, 1.5), confidence: 0.9)
                far.glimpse = true
                builder.observe([far], camera: nil)
            }
        }
        let objects = builder.build().objects
        XCTAssertEqual(objects.map(\.label), ["bed"])
        XCTAssertEqual(objects.first?.size.z ?? 0, 1.5, accuracy: 0.15)
        // A glimpse on nothing starts nothing.
        for _ in 0..<3 {
            var stray = observation("toilet", from: SIMD3(3, 0, 3), to: SIMD3(3.5, 0.4, 3.5))
            stray.glimpse = true
            builder.observe([stray], camera: nil)
        }
        XCTAssertEqual(builder.build().objects.map(\.label), ["bed"])
    }

    func testARugUnderATableIsNotTheTable() {
        let builder = RoomMapBuilder(spec: spec)
        seeFloor(builder)
        for _ in 0..<3 {
            builder.observe([observation("table", from: SIMD3(0, 0.7, 0), to: SIMD3(1.0, 0.75, 0.6)),
                             observation("rug", from: SIMD3(-0.1, 0, -0.1), to: SIMD3(1.1, 0.02, 0.7))], camera: nil)
        }
        let labels = builder.build().objects.map(\.label).sorted()
        XCTAssertEqual(labels, ["rug", "table"])
    }
}
