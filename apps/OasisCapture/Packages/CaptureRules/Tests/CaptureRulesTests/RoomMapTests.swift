import XCTest
import simd
@testable import CaptureRules

final class RoomMapTests: XCTestCase {
    let spec = ObjectSpec.bundled()

    private func cls(_ name: String) -> Int {
        spec.index(of: name)!
    }

    /// Points filling a box on a 4 cm grid.
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

    /// A camera at `position` looking along -z (towards smaller z), landscape 1920x1440.
    private func camera(at position: SIMD3<Float>, yaw: Float = 0) -> PinholeCamera {
        var t = simd_float4x4(simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0)))
        t.columns.3 = SIMD4(position, 1)
        return PinholeCamera(fx: 1500, fy: 1500, cx: 960, cy: 720, width: 1920, height: 1440, transform: t)
    }

    func testTwoBedsApartStayTwoObjects() {
        let builder = RoomMapBuilder(spec: spec)
        for _ in 0..<3 {
            builder.observe([observation("bed", from: SIMD3(0, 0, 0), to: SIMD3(1.4, 0.5, 2.0)),
                             observation("bed", from: SIMD3(3.0, 0, 0), to: SIMD3(4.4, 0.5, 2.0))], camera: nil)
        }
        let beds = builder.build().objects.filter { $0.label == "bed" }
        XCTAssertEqual(beds.count, 2, "\(builder.build().objects.map(\.label))")
        let first = beds.min { $0.min.x < $1.min.x }!
        XCTAssertEqual(first.size.x, 1.4, accuracy: 0.12)
        XCTAssertEqual(first.size.z, 2.0, accuracy: 0.12)
        XCTAssertEqual(first.size.y, 0.5, accuracy: 0.12)
    }

    func testObjectsAppearAfterConfirmationAndStrayOnesAreForgotten() {
        let builder = RoomMapBuilder(spec: spec)
        let table = observation("table", from: SIMD3(0, 0, 0), to: SIMD3(1.0, 0.75, 0.6))
        let first = builder.observe([table], camera: nil)
        XCTAssertEqual(first[0]?.label, "table", "matched (and labelled) from the first frame")
        XCTAssertEqual(first[0]?.placed, false)
        XCTAssertTrue(builder.build().objects.isEmpty, "one observation is not enough to place a box")
        let second = builder.observe([table], camera: nil)
        XCTAssertEqual(second[0]?.placed, true)
        XCTAssertEqual(second[0]?.objectID, first[0]?.objectID)
        XCTAssertEqual(builder.build().objects.map(\.label), ["table"])
        // A one-off detection that never comes back is forgotten.
        _ = builder.observe([observation("chair", from: SIMD3(3, 0, 3), to: SIMD3(3.5, 0.9, 3.5))], camera: nil)
        for _ in 0...spec.tracker.staleObservations { _ = builder.observe([], camera: nil) }
        XCTAssertEqual(builder.tracker.count, 1, "the chair is gone, the table stays")
        builder.reset()
        XCTAssertTrue(builder.build().objects.isEmpty)
    }

    func testLookAlikesAreOneObjectAndTheLabelHoldsUntilAClearLead() {
        let builder = RoomMapBuilder(spec: spec)
        let sofa = observation("sofa", from: SIMD3(0, 0, 0), to: SIMD3(1.8, 0.8, 0.9), confidence: 0.6)
        let armchair = observation("armchair", from: SIMD3(0, 0, 0), to: SIMD3(1.8, 0.8, 0.9), confidence: 0.6)
        for _ in 0..<3 { _ = builder.observe([sofa], camera: nil) }
        XCTAssertEqual(builder.build().objects.map(\.label), ["sofa"])
        let id = builder.build().objects[0].id
        // A couple of armchair frames do not flip it; a run of them does, keeping the identity.
        for _ in 0..<2 {
            let m = builder.observe([armchair], camera: nil)
            XCTAssertEqual(m[0]?.label, "sofa")
            XCTAssertEqual(m[0]?.objectID, id)
        }
        XCTAssertEqual(builder.build().objects.map(\.label), ["sofa"])
        for _ in 0..<6 { _ = builder.observe([armchair], camera: nil) }
        XCTAssertEqual(builder.build().objects.map(\.label), ["armchair"])
        XCTAssertEqual(builder.build().objects[0].id, id)
        XCTAssertEqual(builder.build().objects.count, 1)
    }

    func testAnObjectSeenFromTwoSidesIsOneBoxOfItsFullSize() {
        let builder = RoomMapBuilder(spec: spec)
        // The front half of a 2 m long bed, then (walking round) the back half.
        for _ in 0..<3 { _ = builder.observe([observation("bed", from: SIMD3(0, 0, 0), to: SIMD3(1.4, 0.5, 1.2))], camera: nil) }
        let before = builder.build().objects[0]
        XCTAssertEqual(before.size.z, 1.2, accuracy: 0.12)
        for _ in 0..<3 { _ = builder.observe([observation("bed", from: SIMD3(0, 0, 0.8), to: SIMD3(1.4, 0.5, 2.0))], camera: nil) }
        var after = builder.build().objects
        XCTAssertEqual(after.count, 1, "\(after.map(\.label))")
        XCTAssertEqual(after[0].id, before.id)
        for _ in 0..<10 { _ = builder.observe([observation("bed", from: SIMD3(0, 0, 0.8), to: SIMD3(1.4, 0.5, 2.0))], camera: nil) }
        after = builder.build().objects
        XCTAssertEqual(after[0].size.z, 2.0, accuracy: 0.15, "the box covers both halves")
        XCTAssertEqual(after[0].center.z, 1.0, accuracy: 0.12)
    }

    func testOneBadDepthFrameDoesNotStretchAnEstablishedObject() {
        let builder = RoomMapBuilder(spec: spec)
        let table = observation("table", from: SIMD3(0, 0, 0), to: SIMD3(1.0, 0.75, 0.6))
        for _ in 0..<4 { _ = builder.observe([table], camera: nil) }
        // One frame whose depth put the table's points 2 m further away.
        _ = builder.observe([observation("table", from: SIMD3(0, 0, 0), to: SIMD3(1.0, 0.75, 2.6))], camera: nil)
        for _ in 0..<3 { _ = builder.observe([table], camera: nil) }
        let box = builder.build().objects[0]
        XCTAssertEqual(box.size.z, 0.6, accuracy: 0.15, "voxels hit once do not count")
    }

    func testConfirmedObjectsStayOutOfViewAndGoWhenLookedForAndMissing() {
        let builder = RoomMapBuilder(spec: spec)
        let chair = observation("chair", from: SIMD3(-0.25, 0, -3.25), to: SIMD3(0.25, 0.9, -2.75))
        for _ in 0..<3 { _ = builder.observe([chair], camera: camera(at: SIMD3(0, 1.4, 0))) }
        XCTAssertEqual(builder.build().objects.count, 1)
        // Looking the other way for a long time: it stays.
        for _ in 0..<60 { _ = builder.observe([], camera: camera(at: SIMD3(0, 1.4, 0), yaw: .pi)) }
        XCTAssertEqual(builder.build().objects.count, 1)
        // Standing right on top of it: too close to expect a detection, it stays.
        for _ in 0..<60 { _ = builder.observe([], camera: camera(at: SIMD3(0, 1.4, -2.7))) }
        XCTAssertEqual(builder.build().objects.count, 1)
        // Looking straight at where it should be from 3 m and not finding it: it goes.
        for _ in 0..<spec.tracker.forgetObservations { _ = builder.observe([], camera: camera(at: SIMD3(0, 1.4, 0))) }
        XCTAssertTrue(builder.build().objects.isEmpty)
    }

    func testWallMountedThingsAreLabelledButNotPlaced() {
        let builder = RoomMapBuilder(spec: spec)
        let painting = observation("painting", from: SIMD3(0, 1.2, 0), to: SIMD3(1.0, 1.8, 0.05))
        let lamp = observation("lamp", from: SIMD3(2, 0, 0), to: SIMD3(2.3, 1.5, 0.3))
        for _ in 0..<3 {
            let m = builder.observe([painting, lamp], camera: nil)
            XCTAssertEqual(m[0]?.label, "painting")
            XCTAssertEqual(m[0]?.placed, false)
        }
        XCTAssertEqual(builder.build().objects.map(\.label), ["lamp"])
    }

    func testPeopleTinyAndHugeThingsAreNotTracked() {
        let builder = RoomMapBuilder(spec: spec)
        let person = observation("person", from: SIMD3(1, 0, 1), to: SIMD3(1.5, 1.7, 1.3))
        let few = ObjectObservation(classIndex: cls("lamp"), confidence: 0.9, points: [SIMD3(0, 0, 0), SIMD3(0.1, 0, 0), SIMD3(0.2, 0, 0)])
        let wall = observation("cabinet", from: SIMD3(-3, 0, -3), to: SIMD3(3, 0.3, -2.9))   // 6 m wide
        let m = builder.observe([person, few, wall], camera: nil)
        XCTAssertEqual(m, [nil, nil, nil])
        _ = builder.observe([person, few, wall], camera: nil)
        XCTAssertTrue(builder.build().objects.isEmpty)
        XCTAssertEqual(builder.tracker.count, 0)
    }

    func testTwoHalvesThatGrowTogetherMerge() {
        let builder = RoomMapBuilder(spec: spec)
        // A wardrobe first seen as two separate pieces (left and right doors), then whole.
        let left = observation("wardrobe", from: SIMD3(0, 0, 0), to: SIMD3(0.5, 2.0, 0.6))
        let right = observation("wardrobe", from: SIMD3(1.0, 0, 0), to: SIMD3(1.5, 2.0, 0.6))
        for _ in 0..<3 { _ = builder.observe([left, right], camera: nil) }
        XCTAssertEqual(builder.build().objects.count, 2)
        let whole = observation("wardrobe", from: SIMD3(0, 0, 0), to: SIMD3(1.5, 2.0, 0.6))
        for _ in 0..<10 { _ = builder.observe([whole], camera: nil) }
        let objects = builder.build().objects
        XCTAssertEqual(objects.count, 1, "\(objects.map { "\($0.label) \($0.size)" })")
        XCTAssertEqual(objects[0].size.x, 1.5, accuracy: 0.15)
    }

    func testBoxesTurnWithTheRoomsWalls() {
        let builder = RoomMapBuilder(spec: spec)
        // A wall running 30 degrees off the world x axis.
        let yaw: Float = .pi / 6
        let along = SIMD3(cos(yaw), 0, sin(yaw))
        builder.update(plane: PlaneInfo(id: UUID(), kind: .wall, vertical: true, center: SIMD3(0, 1.2, 0),
                                        xAxis: along, zAxis: SIMD3(0, 1, 0), extent: SIMD2(4, 2.4)))
        // A 2 m x 1 m table lying along that wall.
        let across = SIMD3(-sin(yaw), 0, cos(yaw))
        var pts: [SIMD3<Float>] = []
        var a: Float = 0
        while a <= 2 {
            var b: Float = 0
            while b <= 1 { pts.append(along * a + across * (b + 0.5) + SIMD3(0, 0.7, 0)); b += 0.04 }
            a += 0.04
        }
        let table = ObjectObservation(classIndex: cls("table"), confidence: 0.8, points: pts)
        for _ in 0..<3 { _ = builder.observe([table], camera: nil) }
        let map = builder.build()
        XCTAssertEqual(map.roomYaw!, yaw, accuracy: 0.01)
        let box = map.objects.first { $0.label == "table" }!
        XCTAssertEqual(box.yaw, yaw, accuracy: 0.01)
        XCTAssertEqual(box.size.x, 2.0, accuracy: 0.15, "long side along the wall")
        XCTAssertEqual(box.size.z, 1.0, accuracy: 0.12)
        XCTAssertGreaterThan(box.max.x - box.min.x, 2.0, "world-aligned bounds of a turned box are larger")
        XCTAssertEqual(box.footprint.count, 4)
    }

    func testWallsFoundLaterTurnTheBoxesAlreadyPlaced() {
        let builder = RoomMapBuilder(spec: spec)
        let yaw: Float = .pi / 6
        let along = SIMD3(cos(yaw), 0, sin(yaw)), across = SIMD3(-sin(yaw), 0, cos(yaw))
        var pts: [SIMD3<Float>] = []
        var a: Float = 0
        while a <= 2 {
            var b: Float = 0
            while b <= 1 { pts.append(along * a + across * (b + 0.5) + SIMD3(0, 0.7, 0)); b += 0.04 }
            a += 0.04
        }
        let table = ObjectObservation(classIndex: cls("table"), confidence: 0.8, points: pts)
        for _ in 0..<3 { _ = builder.observe([table], camera: nil) }
        XCTAssertEqual(builder.build().objects[0].yaw, 0, "no walls yet: world-aligned")
        builder.update(plane: PlaneInfo(id: UUID(), kind: .wall, vertical: true, center: SIMD3(0, 1.2, 0),
                                        xAxis: along, zAxis: SIMD3(0, 1, 0), extent: SIMD2(4, 2.4)))
        _ = builder.observe([], camera: nil)
        let box = builder.build().objects[0]
        XCTAssertEqual(box.yaw, yaw, accuracy: 0.01)
        XCTAssertEqual(box.size.x, 2.0, accuracy: 0.15)
    }

    func testWallsFromVerticalPlanesAndBounds() {
        var map = RoomMap()
        map.planes = [
            PlaneInfo(id: UUID(), kind: .wall, vertical: true, center: SIMD3(2, 1.2, 0), xAxis: SIMD3(1, 0, 0),
                      zAxis: SIMD3(0, 1, 0), extent: SIMD2(4, 2.4)),
            PlaneInfo(id: UUID(), kind: .floor, vertical: false, center: SIMD3(2, 0, 2), xAxis: SIMD3(1, 0, 0),
                      zAxis: SIMD3(0, 0, 1), extent: SIMD2(4, 4)),
            PlaneInfo(id: UUID(), kind: .window, vertical: true, center: SIMD3(0, 1.5, 2), xAxis: SIMD3(0, 0, 1),
                      zAxis: SIMD3(0, 1, 0), extent: SIMD2(1, 1)),
        ]
        let walls = map.walls
        XCTAssertEqual(walls.count, 1, "windows are not walls")
        XCTAssertEqual(walls[0].from, SIMD2(0, 0))
        XCTAssertEqual(walls[0].to, SIMD2(4, 0))
        XCTAssertEqual(walls[0].height, 2.4)
        XCTAssertEqual(map.floors.count, 1)
        let bounds = map.bounds!
        XCTAssertEqual(bounds.min, SIMD2(0, 0))
        XCTAssertEqual(bounds.max, SIMD2(4, 4))
    }

    func testLabelMemorySteadiesFlickerWithinAFamily() {
        let detection = DetectionSpec.bundled()
        var memory = LabelMemory(spec: detection)
        func id(_ name: String) -> Int32 { Int32(detection.classes.first { $0.name == name }!.id) }
        let sofa = id("sofa"), chair = id("armchair"), floor = id("floor")
        func result(_ cls: Int32) -> SegmentationResult {
            var classes = [Int32](repeating: floor, count: 64 * 64)
            for y in 10..<40 { for x in 10..<40 { classes[y * 64 + x] = cls } }
            return OutlineExtractor.extract(classes: classes, width: 64, height: 64, spec: detection)
        }
        for _ in 0..<4 { _ = memory.steady(result(sofa)) }
        let flicker = memory.steady(result(chair))
        XCTAssertEqual(flicker.regions.first { $0.group == "furniture" }?.label, "sofa")
        var latest = flicker
        for _ in 0..<6 { latest = memory.steady(result(chair)) }
        XCTAssertEqual(latest.regions.first { $0.group == "furniture" }?.label, "armchair")
        XCTAssertEqual(latest.regions.first { $0.group == "structure" }?.label, "floor")
    }
}
