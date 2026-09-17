import XCTest
import simd
@testable import CaptureRules

final class RoomMapTests: XCTestCase {
    let spec = DetectionSpec.bundled()

    private func classId(_ name: String) -> Int {
        spec.classes.first { $0.name == name }!.id
    }

    /// Points filling a box on a 5 cm grid, `passes` times (a voxel needs three votes).
    private func fill(_ builder: RoomMapBuilder, _ name: String, from lo: SIMD3<Float>, to hi: SIMD3<Float>, passes: Int = 3) {
        let id = classId(name)
        for _ in 0..<passes {
            var x = lo.x
            while x <= hi.x {
                var y = lo.y
                while y <= hi.y {
                    var z = lo.z
                    while z <= hi.z {
                        builder.add(point: SIMD3(x, y, z), classId: id)
                        z += 0.05
                    }
                    y += 0.05
                }
                x += 0.05
            }
        }
    }

    /// Builds until objects are confirmed.
    private func settled(_ builder: RoomMapBuilder) -> RoomMap {
        var map = RoomMap()
        for _ in 0..<spec.confirmBuilds { map = builder.build() }
        return map
    }

    func testTwoBedsApartStayTwoObjects() {
        let builder = RoomMapBuilder(spec: spec)
        fill(builder, "bed", from: SIMD3(0, 0, 0), to: SIMD3(1.4, 0.5, 2.0))
        fill(builder, "bed", from: SIMD3(3.0, 0, 0), to: SIMD3(4.4, 0.5, 2.0))
        let map = settled(builder)
        let beds = map.objects.filter { $0.label == "bed" }
        XCTAssertEqual(beds.count, 2, "\(map.objects.map(\.label))")
        let first = beds.min { $0.min.x < $1.min.x }!
        XCTAssertEqual(first.size.x, 1.4, accuracy: 0.1)
        XCTAssertEqual(first.size.z, 2.0, accuracy: 0.1)
        XCTAssertEqual(first.size.y, 0.5, accuracy: 0.1)
    }

    func testTallWardrobeSeenInPiecesIsOneObject() {
        let builder = RoomMapBuilder(spec: spec)
        fill(builder, "wardrobe", from: SIMD3(0, 0, 0), to: SIMD3(1.2, 0.3, 0.6))
        fill(builder, "wardrobe", from: SIMD3(0, 1.8, 0), to: SIMD3(1.2, 2.1, 0.6))
        let wardrobes = settled(builder).objects.filter { $0.label == "wardrobe" }
        XCTAssertEqual(wardrobes.count, 1)
        XCTAssertEqual(wardrobes[0].size.y, 2.1, accuracy: 0.15)
    }

    func testLookAlikesVoteAsOneObjectAndKeepTheirLabel() {
        let builder = RoomMapBuilder(spec: spec)
        // A sofa the model calls an armchair on a third of its points: one object, called sofa.
        fill(builder, "sofa", from: SIMD3(0, 0, 0), to: SIMD3(1.8, 0.8, 0.9), passes: 2)
        fill(builder, "armchair", from: SIMD3(0, 0, 0), to: SIMD3(1.8, 0.8, 0.9), passes: 1)
        var map = settled(builder)
        XCTAssertEqual(map.objects.map(\.label), ["sofa"])
        let id = map.objects[0].id
        // A little more armchair does not flip the label; a clear lead does.
        fill(builder, "armchair", from: SIMD3(0, 0, 0), to: SIMD3(1.8, 0.8, 0.9), passes: 1)
        map = builder.build()
        XCTAssertEqual(map.objects.map(\.label), ["sofa"])
        XCTAssertEqual(map.objects[0].id, id, "the object keeps its identity")
        fill(builder, "armchair", from: SIMD3(0, 0, 0), to: SIMD3(1.8, 0.8, 0.9), passes: 3)
        map = builder.build()
        XCTAssertEqual(map.objects.map(\.label), ["armchair"])
        XCTAssertEqual(map.objects[0].id, id)
    }

    func testObjectsAppearAfterConfirmationAndLingerWhenUnseen() {
        let builder = RoomMapBuilder(spec: spec)
        fill(builder, "table", from: SIMD3(0, 0, 0), to: SIMD3(1.0, 0.75, 0.6))
        XCTAssertTrue(builder.build().objects.isEmpty, "one build is not enough to show a box")
        XCTAssertEqual(builder.build().objects.map(\.label), ["table"])
        builder.reset()
        XCTAssertTrue(builder.build().objects.isEmpty, "reset forgets everything")
        // A box whose votes vanish stays for graceBuilds builds, then goes.
        fill(builder, "table", from: SIMD3(0, 0, 0), to: SIMD3(1.0, 0.75, 0.6))
        _ = settled(builder)
        builder.forgetVotes()
        for _ in 0..<spec.graceBuilds { XCTAssertEqual(builder.build().objects.count, 1) }
        XCTAssertTrue(builder.build().objects.isEmpty)
    }

    func testExtentGrowsSmoothlyAndIdentityHolds() {
        let builder = RoomMapBuilder(spec: spec)
        fill(builder, "bed", from: SIMD3(0, 0, 0), to: SIMD3(1.0, 0.5, 1.0))
        let before = settled(builder).objects[0]
        // The rest of the bed comes into view: the box grows towards it, not all at once.
        fill(builder, "bed", from: SIMD3(1.0, 0, 0), to: SIMD3(1.5, 0.5, 2.0))
        let after = builder.build().objects[0]
        XCTAssertEqual(after.id, before.id)
        XCTAssertGreaterThan(after.size.z, before.size.z)
        XCTAssertLessThan(after.size.z, 2.0)
        var latest = after
        for _ in 0..<12 { latest = builder.build().objects[0] }
        XCTAssertEqual(latest.size.z, 2.0, accuracy: 0.15, "and settles on the full size")
        XCTAssertEqual(latest.id, before.id)
    }

    func testStrayLabelsWallsPeopleAndHugeThingsAreNotBoxed() {
        let builder = RoomMapBuilder(spec: spec)
        for i in 0..<5 { builder.add(point: SIMD3(Float(i) * 0.3, 0, 0), classId: classId("lamp")) }   // too few
        fill(builder, "wall", from: SIMD3(0, 0, 3), to: SIMD3(3, 2.4, 3.05))
        fill(builder, "person", from: SIMD3(1, 0, 1), to: SIMD3(1.5, 1.7, 1.3))
        fill(builder, "apparel", from: SIMD3(2, 0, 1), to: SIMD3(2.5, 0.3, 1.3))
        fill(builder, "cabinet", from: SIMD3(-3, 0, -3), to: SIMD3(3, 0.3, -2.9))   // 6 m wide: not furniture
        XCTAssertTrue(settled(builder).objects.isEmpty, "\(settled(builder).objects.map(\.label))")
    }

    func testBedCalledSofaOnOneSideIsOneBoxNotTwo() {
        let builder = RoomMapBuilder(spec: spec)
        // The model says bed on the left half and sofa on the right half.
        fill(builder, "bed", from: SIMD3(0, 0, 0), to: SIMD3(0.9, 0.5, 2.0))
        fill(builder, "sofa", from: SIMD3(1.0, 0, 0), to: SIMD3(1.8, 0.5, 2.0))
        let map = settled(builder)
        XCTAssertEqual(map.objects.count, 1, "\(map.objects.map(\.label))")
        XCTAssertEqual(map.objects[0].size.x, 1.8, accuracy: 0.2)
    }

    func testWallMountedThingsAreNotBoxed() {
        let builder = RoomMapBuilder(spec: spec)
        fill(builder, "painting", from: SIMD3(0, 1.2, 0), to: SIMD3(1.0, 1.8, 0.05))
        fill(builder, "mirror", from: SIMD3(2, 1.0, 0), to: SIMD3(2.6, 1.8, 0.05))
        fill(builder, "curtain", from: SIMD3(3, 0, 0), to: SIMD3(4.0, 2.2, 0.1))
        fill(builder, "lamp", from: SIMD3(5, 0, 0), to: SIMD3(5.3, 1.5, 0.3))
        XCTAssertEqual(settled(builder).objects.map(\.label), ["lamp"])
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
        let id = classId("table")
        for _ in 0..<3 {
            var a: Float = 0
            while a <= 2 {
                var b: Float = 0
                while b <= 1 {
                    builder.add(point: along * a + across * (b + 0.5) + SIMD3(0, 0.7, 0), classId: id)
                    b += 0.05
                }
                a += 0.05
            }
        }
        let map = settled(builder)
        XCTAssertEqual(map.roomYaw!, yaw, accuracy: 0.01)
        let table = map.objects.first { $0.label == "table" }!
        XCTAssertEqual(table.yaw, yaw, accuracy: 0.01)
        XCTAssertEqual(table.size.x, 2.0, accuracy: 0.2, "long side along the wall (8 cm voxels pad the ends)")
        XCTAssertEqual(table.size.z, 1.0, accuracy: 0.15)
        // World-aligned bounds of a turned box are larger than the box itself.
        XCTAssertGreaterThan(table.max.x - table.min.x, 2.0)
        XCTAssertEqual(table.footprint.count, 4)
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
        var memory = LabelMemory(spec: spec)
        let sofa = Int32(classId("sofa")), chair = Int32(classId("armchair")), floor = Int32(classId("floor"))
        func result(_ cls: Int32) -> SegmentationResult {
            var classes = [Int32](repeating: floor, count: 64 * 64)
            for y in 10..<40 { for x in 10..<40 { classes[y * 64 + x] = cls } }
            return OutlineExtractor.extract(classes: classes, width: 64, height: 64, spec: spec)
        }
        for _ in 0..<4 { _ = memory.steady(result(sofa)) }
        // One frame that says armchair is shown as sofa; a run of them is not.
        let flicker = memory.steady(result(chair))
        XCTAssertEqual(flicker.regions.first { $0.group == "furniture" }?.label, "sofa")
        var latest = flicker
        for _ in 0..<6 { latest = memory.steady(result(chair)) }
        XCTAssertEqual(latest.regions.first { $0.group == "furniture" }?.label, "armchair")
        // Memory never turns furniture into floor: the region's own family wins.
        let floorRegion = latest.regions.first { $0.group == "structure" }
        XCTAssertEqual(floorRegion?.label, "floor")
    }
}
