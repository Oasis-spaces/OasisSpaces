import XCTest
import simd
@testable import CaptureRules

final class RoomMapTests: XCTestCase {
    let spec = DetectionSpec.bundled()

    private func classId(_ name: String) -> Int {
        spec.classes.first { $0.name == name }!.id
    }

    /// Points filling a box, on a 5 cm grid, each seen twice (a voxel needs two votes).
    private func fill(_ builder: RoomMapBuilder, _ name: String, from lo: SIMD3<Float>, to hi: SIMD3<Float>) {
        let id = classId(name)
        for _ in 0..<2 {
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

    func testTwoBedsApartStayTwoObjects() {
        let builder = RoomMapBuilder(spec: spec)
        fill(builder, "bed", from: SIMD3(0, 0, 0), to: SIMD3(1.4, 0.5, 2.0))
        fill(builder, "bed", from: SIMD3(3.0, 0, 0), to: SIMD3(4.4, 0.5, 2.0))
        let map = builder.build()
        let beds = map.objects.filter { $0.label == "bed" }
        XCTAssertEqual(beds.count, 2, "\(map.objects.map(\.label))")
        let first = beds.min { $0.min.x < $1.min.x }!
        XCTAssertEqual(first.size.x, 1.4, accuracy: 0.1)
        XCTAssertEqual(first.size.z, 2.0, accuracy: 0.1)
        XCTAssertEqual(first.size.y, 0.5, accuracy: 0.1)
    }

    func testTallWardrobeSeenInPiecesIsOneObject() {
        let builder = RoomMapBuilder(spec: spec)
        // Its base and its top, nothing in between (the middle was never labelled).
        fill(builder, "wardrobe", from: SIMD3(0, 0, 0), to: SIMD3(1.2, 0.3, 0.6))
        fill(builder, "wardrobe", from: SIMD3(0, 1.8, 0), to: SIMD3(1.2, 2.1, 0.6))
        let map = builder.build()
        let wardrobes = map.objects.filter { $0.label == "wardrobe" }
        XCTAssertEqual(wardrobes.count, 1)
        XCTAssertEqual(wardrobes[0].size.y, 2.1, accuracy: 0.15)
    }

    func testStrayLabelsWallsAndPeopleAreNotBoxed() {
        let builder = RoomMapBuilder(spec: spec)
        for i in 0..<5 { builder.add(point: SIMD3(Float(i) * 0.3, 0, 0), classId: classId("lamp")) }   // too few
        fill(builder, "wall", from: SIMD3(0, 0, 3), to: SIMD3(3, 2.4, 3.05))
        fill(builder, "person", from: SIMD3(1, 0, 1), to: SIMD3(1.5, 1.7, 1.3))
        fill(builder, "apparel", from: SIMD3(2, 0, 1), to: SIMD3(2.5, 0.3, 1.3))
        XCTAssertTrue(builder.build().objects.isEmpty, "\(builder.build().objects.map(\.label))")
    }

    func testMajorityClassWins() {
        let builder = RoomMapBuilder(spec: spec)
        fill(builder, "sofa", from: SIMD3(0, 0, 0), to: SIMD3(1.8, 0.8, 0.9))
        // One pass of a wrong label over the same place loses to two of the right one.
        let wrong = classId("bed")
        var x: Float = 0
        while x <= 1.8 { builder.add(point: SIMD3(x, 0.4, 0.4), classId: wrong); x += 0.05 }
        let labels = builder.build().objects.map(\.label)
        XCTAssertEqual(labels, ["sofa"])
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
}
