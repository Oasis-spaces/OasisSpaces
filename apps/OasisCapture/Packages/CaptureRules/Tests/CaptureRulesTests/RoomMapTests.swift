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
        // The front half of a 2 m long bed, looked at for a moment or two, then (walking round) the back half.
        for _ in 0..<5 { _ = builder.observe([observation("bed", from: SIMD3(0, 0, 0), to: SIMD3(1.4, 0.5, 1.2))], camera: nil) }
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

    func testAnExtentCoversEverythingConnectedAndStopsAtAGap() {
        // A bed's top seen a lot, its side seen a little, and a mask's edge that landed on the wall a metre behind.
        var hits: [(centre: SIMD3<Float>, hits: Int)] = []
        var x: Float = 0.025
        while x < 2 {
            var z: Float = 0.025
            while z < 1.4 { hits.append((SIMD3(x, 0.475, z), 20)); z += 0.05 }   // the top, busy
            var y: Float = 0.025
            while y < 0.45 { hits.append((SIMD3(x, y, 0.025), 2)); y += 0.05 }   // one side, thin
            x += 0.05
        }
        x = 0.025
        while x < 0.4 { hits.append((SIMD3(x, 0.475, 2.525), 3)); x += 0.05 }    // the wall behind, beyond a gap
        let bounds = ObjectTracker.hitBounds(hits, yaw: 0, voxel: 0.05, binShare: 0.05, gap: 0.15)!
        XCTAssertEqual(bounds.lo.y, 0, accuracy: 0.051, "the side counts, though the top was seen ten times more")
        XCTAssertEqual(bounds.hi.y, 0.5, accuracy: 0.051)
        XCTAssertEqual(bounds.hi.z, 1.4, accuracy: 0.051, "what lies beyond a gap is not the object")
        XCTAssertEqual(bounds.hi.x - bounds.lo.x, 2.0, accuracy: 0.051)
    }

    func testFurnitureReachesTheFloorAndAWardrobeTheWallBehindIt() {
        let builder = RoomMapBuilder(spec: spec)
        builder.update(plane: PlaneInfo(id: UUID(), kind: .floor, vertical: false, center: SIMD3(0, 0, 0), xAxis: SIMD3(1, 0, 0),
                                        zAxis: SIMD3(0, 0, 1), extent: SIMD2(4, 4)))
        builder.update(plane: PlaneInfo(id: UUID(), kind: .wall, vertical: true, center: SIMD3(0, 1.2, 2), xAxis: SIMD3(1, 0, 0),
                                        zAxis: SIMD3(0, 1, 0), extent: SIMD2(4, 2.4)))
        // Only the wardrobe's doors are seen (0.6 m in front of the wall), from knee height up;
        // a bed's top from 0.3 m up; a wall cabinet at head height; a table half a metre off the wall.
        let doors = observation("wardrobe", from: SIMD3(-0.5, 0.3, 1.36), to: SIMD3(0.5, 2.0, 1.4))
        let bed = observation("bed", from: SIMD3(-1.9, 0.3, -1.5), to: SIMD3(-0.5, 0.5, 0.4))
        let cabinet = observation("cabinet", from: SIMD3(1.2, 1.5, 1.7), to: SIMD3(1.9, 2.1, 1.95))
        let table = observation("table", from: SIMD3(0.9, 0.7, 0.5), to: SIMD3(1.9, 0.75, 1.5))
        for _ in 0..<3 { _ = builder.observe([doors, bed, cabinet, table], camera: nil) }
        let objects = builder.build().objects
        let wardrobe = objects.first { $0.label == "wardrobe" }!
        XCTAssertEqual(wardrobe.min.y, 0, accuracy: 0.01, "down to the floor")
        XCTAssertEqual(wardrobe.max.z, 2.0, accuracy: 0.01, "back to the wall")
        XCTAssertEqual(wardrobe.size.z, 0.65, accuracy: 0.06)
        XCTAssertEqual(wardrobe.size.x, 1.0, accuracy: 0.11, "its width is what was seen")
        let placedBed = objects.first { $0.label == "bed" }!
        XCTAssertEqual(placedBed.min.y, 0, accuracy: 0.01)
        XCTAssertEqual(placedBed.size.y, 0.5, accuracy: 0.06)
        let wallCabinet = objects.first { $0.label == "cabinet" }!
        XCTAssertEqual(wallCabinet.min.y, 1.5, accuracy: 0.06, "a cabinet on the wall stays on the wall")
        XCTAssertEqual(wallCabinet.max.z, 2.0, accuracy: 0.06)
        let placedTable = objects.first { $0.label == "table" }!
        XCTAssertEqual(placedTable.max.z, 1.5, accuracy: 0.06, "a table half a metre off the wall is not pushed against it")
        XCTAssertEqual(placedTable.min.y, 0, accuracy: 0.01, "a table stands on the floor, though only its top was seen")
    }

    func testACabinetAboveADeskIsNotTheDeskAndASwitchPlateIsNotPlaced() {
        let builder = RoomMapBuilder(spec: spec)
        let desk = observation("desk", from: SIMD3(0, 0, 0), to: SIMD3(1.2, 0.75, 0.6))
        let cabinet = observation("cabinet", from: SIMD3(0, 1.5, 0), to: SIMD3(1.2, 2.1, 0.35))
        let plate = observation("heater", from: SIMD3(2, 1.2, 0), to: SIMD3(2.16, 1.28, 0.04))
        for _ in 0..<4 { _ = builder.observe([desk, cabinet, plate], camera: nil) }
        let objects = builder.build().objects
        XCTAssertEqual(Set(objects.map(\.label)), ["desk", "cabinet"], "\(objects.map { "\($0.label) \($0.size)" })")
        XCTAssertEqual(objects.first { $0.label == "desk" }!.size.y, 0.75, accuracy: 0.11)
    }

    func testThePartsOfAWardrobeSeenOneDoorAtATimeAreOneWardrobe() {
        let builder = RoomMapBuilder(spec: spec)
        // Three doors, each seen on its own, with a hand's width of frame between them.
        for door in 0..<3 {
            let x = Float(door) * 0.6
            for _ in 0..<(door == 2 ? 12 : 5) { _ = builder.observe([observation("wardrobe", from: SIMD3(x, 0.1, 0), to: SIMD3(x + 0.45, 2.0, 0.08))], camera: nil) }
        }
        let objects = builder.build().objects
        XCTAssertEqual(objects.count, 1, "\(objects.map { "\($0.label) \($0.size)" })")
        XCTAssertEqual(objects[0].size.x, 1.65, accuracy: 0.16)
        // Two chairs side by side stay two chairs.
        let chairs = RoomMapBuilder(spec: spec)
        for _ in 0..<3 {
            _ = chairs.observe([observation("chair", from: SIMD3(0, 0, 0), to: SIMD3(0.45, 0.9, 0.45)),
                                observation("chair", from: SIMD3(0.6, 0, 0), to: SIMD3(1.05, 0.9, 0.45))], camera: nil)
        }
        XCTAssertEqual(chairs.build().objects.count, 2)
    }

    func testWhatLandsUnderTheFloorOrBehindAWallIsNotPlaced() {
        let builder = RoomMapBuilder(spec: spec)
        builder.update(plane: PlaneInfo(id: UUID(), kind: .floor, vertical: false, center: SIMD3(0, 0, 0), xAxis: SIMD3(1, 0, 0),
                                        zAxis: SIMD3(0, 0, 1), extent: SIMD2(4, 4)))
        builder.update(plane: PlaneInfo(id: UUID(), kind: .wall, vertical: true, center: SIMD3(0, 1.2, -2), xAxis: SIMD3(1, 0, 0),
                                        zAxis: SIMD3(0, 1, 0), extent: SIMD2(4, 2.4)))
        let eye = camera(at: SIMD3(0, 1.4, 1))
        // Depth gone wrong: a chest of drawers under the floor, a wardrobe two metres behind the wall.
        let sunk = observation("chest of drawers", from: SIMD3(-1, -1.6, -1), to: SIMD3(0, -0.4, -0.6))
        let beyond = observation("wardrobe", from: SIMD3(0.2, 0, -4.5), to: SIMD3(1.2, 2, -4))
        // A desk against the wall whose mask's edge ran 0.4 m through it: the desk stays, cut at the wall.
        let desk = observation("desk", from: SIMD3(-1.9, 0, -2.4), to: SIMD3(-0.9, 0.75, -1.4))
        for _ in 0..<4 { _ = builder.observe([sunk, beyond, desk], camera: eye) }
        let objects = builder.build().objects
        XCTAssertEqual(objects.map(\.label), ["desk"])
        XCTAssertEqual(objects[0].min.z, -2.0, accuracy: 0.06, "furniture stops at the wall")
        XCTAssertEqual(objects[0].max.z, -1.4, accuracy: 0.06)
    }

    func testTwoSightingsOfAScreenAHandApartAreOneScreen() {
        let builder = RoomMapBuilder(spec: spec)
        let near = observation("television", from: SIMD3(0, 0.9, 0), to: SIMD3(0.6, 1.25, 0.04))
        let far = observation("television", from: SIMD3(0.04, 0.9, 0.12), to: SIMD3(0.64, 1.25, 0.16))
        for _ in 0..<3 { _ = builder.observe([near], camera: nil) }
        for _ in 0..<3 { _ = builder.observe([far], camera: nil) }
        XCTAssertEqual(builder.build().objects.count, 1)
    }

    func testWhatRestsOnFurnitureStaysThereAndAWardrobeIsNoDeeperThanWardrobesAre() {
        let builder = RoomMapBuilder(spec: spec)
        builder.update(plane: PlaneInfo(id: UUID(), kind: .floor, vertical: false, center: SIMD3(0, 0, 0), xAxis: SIMD3(1, 0, 0),
                                        zAxis: SIMD3(0, 0, 1), extent: SIMD2(4, 4)))
        builder.update(plane: PlaneInfo(id: UUID(), kind: .wall, vertical: true, center: SIMD3(0, 1.2, 2), xAxis: SIMD3(1, 0, 0),
                                        zAxis: SIMD3(0, 1, 0), extent: SIMD2(4, 2.4)))
        // A box lying on a stool; a wardrobe on the wall with a chair's worth of clutter in front taken for part of it.
        let stool = observation("stool", from: SIMD3(-1.5, 0, 0), to: SIMD3(-1.0, 0.45, 0.5))
        let box = observation("box", from: SIMD3(-1.4, 0.47, 0.1), to: SIMD3(-1.1, 0.75, 0.4))
        let wardrobe = observation("wardrobe", from: SIMD3(0, 0, 1.0), to: SIMD3(1.6, 2.0, 1.45))
        for _ in 0..<3 { _ = builder.observe([stool, box, wardrobe], camera: nil) }
        let objects = builder.build().objects
        let placedBox = objects.first { $0.label == "box" }!
        XCTAssertEqual(placedBox.min.y, 0.45, accuracy: 0.06, "the box is on the stool, not a column down to the floor")
        let placedWardrobe = objects.first { $0.label == "wardrobe" }!
        XCTAssertEqual(placedWardrobe.max.z, 2.0, accuracy: 0.01)
        XCTAssertEqual(placedWardrobe.size.z, spec.tracker.unitDepthMetres, accuracy: 0.01)
    }

    func testFurnitureOnBareWallIsTheDetectorSeeingThings() {
        // A surface map whose left half is wall and right half is wardrobe.
        let wall = Int32(0), wardrobe = Int32(35)
        var classes = [Int32](repeating: wall, count: 64 * 64)
        for y in 0..<64 { for x in 32..<64 { classes[y * 64 + x] = wardrobe } }
        func instance(_ name: String, columns: Range<Int>) -> Instance {
            var mask = [UInt8](repeating: 0, count: 32 * 32)
            for y in 4..<28 { for x in columns { mask[y * 32 + x] = 1 } }
            return Instance(classIndex: cls(name), confidence: 0.6, minX: 0, minY: 0, maxX: 1, maxY: 1, mask: mask,
                            maskWidth: 32, maskHeight: 32, area: 24 * columns.count)
        }
        func seeingThings(_ i: Instance) -> Bool { spec.isOnBareSurface(i, bare: [wall], classes: classes, width: 64, height: 64) }
        XCTAssertEqual(instance("refrigerator", columns: 2..<14).share(on: [wall], classes: classes, width: 64, height: 64), 1, accuracy: 0.01)
        XCTAssertTrue(seeingThings(instance("refrigerator", columns: 2..<14)), "a fridge that is a stretch of wall")
        XCTAssertTrue(seeingThings(instance("window", columns: 2..<14)))
        XCTAssertFalse(seeingThings(instance("wardrobe", columns: 18..<30)), "furniture where the surface model sees furniture")
        XCTAssertFalse(seeingThings(instance("wardrobe", columns: 12..<30)), "partly on wall is not on bare wall")
        XCTAssertFalse(seeingThings(instance("television", columns: 2..<14)), "a screen hangs on a wall")
        XCTAssertFalse(seeingThings(instance("door", columns: 2..<14)), "a door is often wall to the surface model")
        XCTAssertFalse(seeingThings(instance("rug", columns: 2..<14)))
    }

    /// A wall along x at z = 2 (so the room's direction is known), and optionally one along z at x = 2 and the floor.
    private func room(_ builder: RoomMapBuilder, secondWall: Bool = false, floor: Bool = false) {
        builder.update(plane: PlaneInfo(id: UUID(), kind: .wall, vertical: true, center: SIMD3(0.5, 1.2, 2), xAxis: SIMD3(1, 0, 0),
                                        zAxis: SIMD3(0, 1, 0), extent: SIMD2(5, 2.4)))
        if secondWall {
            builder.update(plane: PlaneInfo(id: UUID(), kind: .wall, vertical: true, center: SIMD3(2, 1.2, 0.5), xAxis: SIMD3(0, 0, 1),
                                            zAxis: SIMD3(0, 1, 0), extent: SIMD2(3, 2.4)))
        }
        if floor {
            builder.update(plane: PlaneInfo(id: UUID(), kind: .floor, vertical: false, center: SIMD3(0, 0, 0), xAxis: SIMD3(1, 0, 0),
                                            zAxis: SIMD3(0, 0, 1), extent: SIMD2(6, 6)))
        }
    }

    func testAGlimpseOfSomethingWrongDoesNotStickAndASecondLookDoes() {
        let builder = RoomMapBuilder(spec: spec)
        let cabinet = observation("cabinet", from: SIMD3(0, 1.4, 0), to: SIMD3(0.6, 1.95, 0.3))
        // For two frames running the detector also takes the door below and beside it for a wardrobe.
        let wrong = observation("wardrobe", from: SIMD3(0, 0.1, 0), to: SIMD3(1.4, 1.9, 0.3))
        for _ in 0..<4 { _ = builder.observe([cabinet], camera: nil) }
        for _ in 0..<2 { _ = builder.observe([cabinet, wrong], camera: nil) }
        for _ in 0..<8 { _ = builder.observe([cabinet], camera: nil) }
        var objects = builder.build().objects
        XCTAssertEqual(objects.map(\.label), ["cabinet"])
        XCTAssertEqual(objects[0].min.y, 1.4, accuracy: 0.06, "two frames running are one glimpse")
        XCTAssertEqual(objects[0].size.x, 0.6, accuracy: 0.11)
        // Seen again on a later pass, it is there after all.
        for _ in 0..<2 { _ = builder.observe([cabinet, wrong], camera: nil) }
        for _ in 0..<8 { _ = builder.observe([cabinet], camera: nil) }
        objects = builder.build().objects
        XCTAssertEqual(objects.count, 1)
        XCTAssertEqual(objects[0].size.x, 1.4, accuracy: 0.15)
    }

    func testAWardrobeTakesThePatternedDoorsTheDetectorCallsCurtains() {
        let builder = RoomMapBuilder(spec: spec)
        room(builder)
        // The patterned sliding door, seen first and called a curtain; then the open section, called a wardrobe.
        let door = observation("curtain", from: SIMD3(0.5, 0.1, 1.36), to: SIMD3(1.4, 2.0, 1.4))
        let body = observation("wardrobe", from: SIMD3(0, 0.1, 1.36), to: SIMD3(0.7, 2.0, 1.4))
        // A real curtain at the wall beside it, and a cloth hung next to it in the plane of its doors.
        let curtain = observation("curtain", from: SIMD3(1.5, 0.3, 1.94), to: SIMD3(2.4, 2.2, 1.98))
        let cloth = observation("curtain", from: SIMD3(-0.5, 0.5, 1.36), to: SIMD3(-0.1, 1.8, 1.4))
        for _ in 0..<2 { _ = builder.observe([door, curtain, cloth], camera: nil) }
        XCTAssertTrue(builder.build().objects.isEmpty, "curtains are outlined, not placed")
        var matches: [ObservationMatch?] = []
        for _ in 0..<10 { matches = builder.observe([body, door, curtain, cloth], camera: nil) }   // (the box eases to its size)
        let objects = builder.build().objects
        XCTAssertEqual(objects.map(\.label), ["wardrobe"])
        XCTAssertEqual(objects[0].min.x, 0, accuracy: 0.06)
        XCTAssertEqual(objects[0].max.x, 1.4, accuracy: 0.11, "the wardrobe is as wide as its body and its door")
        XCTAssertEqual(matches[1]?.label, "wardrobe", "the door is outlined as the wardrobe it belongs to")
        XCTAssertEqual(matches[2]?.label, "curtain", "a curtain at the wall, a wardrobe's depth behind its front, is a curtain")
        XCTAssertEqual(matches[3]?.label, "curtain", "a cloth hung beside the wardrobe does not lie over it")
    }

    func testBareWallWithANameIsNothingUnlessItIsAWardrobesDoor() {
        let builder = RoomMapBuilder(spec: spec)
        room(builder)
        let body = observation("wardrobe", from: SIMD3(0, 0.1, 1.36), to: SIMD3(0.7, 2.0, 1.4))
        // The surface model saw bare wall in both; one lies over the wardrobe's front, the other nowhere near.
        let panel = ObjectObservation(classIndex: cls("refrigerator"), confidence: 0.6,
                                      points: points(from: SIMD3(0.5, 0.1, 1.36), to: SIMD3(1.4, 2.0, 1.4)), doubtful: true)
        let elsewhere = ObjectObservation(classIndex: cls("refrigerator"), confidence: 0.6,
                                          points: points(from: SIMD3(-2, 0.1, 0), to: SIMD3(-1.96, 2.0, 0.9)), doubtful: true)
        var matches = builder.observe([panel, elsewhere], camera: nil)
        XCTAssertEqual(matches, [nil, nil])
        XCTAssertEqual(builder.tracker.count, 0, "nothing is kept of it")
        for _ in 0..<3 { matches = builder.observe([body, panel, elsewhere], camera: nil) }
        XCTAssertEqual(matches[1]?.label, "wardrobe")
        XCTAssertNil(matches[2])
        XCTAssertEqual(builder.build().objects.map(\.label), ["wardrobe"])
        XCTAssertEqual(builder.tracker.count, 1)
    }

    func testTwoUnitsThatMeetInACornerOfTheRoomAreTwo() {
        let builder = RoomMapBuilder(spec: spec)
        room(builder, secondWall: true)
        // A wardrobe's doors along one wall, a cabinet's front along the next, a hand apart in the corner.
        let wardrobe = observation("wardrobe", from: SIMD3(0, 0.1, 1.36), to: SIMD3(1.7, 2.0, 1.4))
        let cabinet = observation("cabinet", from: SIMD3(1.66, 1.3, 0.4), to: SIMD3(1.7, 1.9, 1.3))
        for _ in 0..<5 { _ = builder.observe([wardrobe, cabinet], camera: nil) }
        let objects = builder.build().objects
        XCTAssertEqual(Set(objects.map(\.label)), ["wardrobe", "cabinet"], "\(objects.map { "\($0.label) \($0.size)" })")
    }

    func testADeskWhoseTopIsKneeHighIsNotADesk() {
        let builder = RoomMapBuilder(spec: spec)
        room(builder, floor: true)
        // A padded stool the detector calls a desk, and a desk.
        let stool = observation("desk", from: SIMD3(-2, 0.1, -1), to: SIMD3(-1.5, 0.45, -0.5))
        let desk = observation("desk", from: SIMD3(0, 0.1, -1), to: SIMD3(1.2, 0.74, -0.4))
        for _ in 0..<4 { _ = builder.observe([stool, desk], camera: nil) }
        let objects = builder.build().objects
        XCTAssertEqual(objects.count, 1)
        XCTAssertEqual(objects[0].min.x, 0, accuracy: 0.06)
        XCTAssertEqual(objects[0].top ?? 0, 0.75, accuracy: 0.051)
    }

    func testAThingIsOutlinedFromItsSecondSightingUnderItsMostVotedName() {
        var memory = SightingMemory(spec: spec)
        let eye = SIMD3<Float>(0, 1.4, 0)
        func sighting(_ name: String, _ confidence: Float, at degrees: Float, radius: Float = 0.2) -> Sighting {
            let a = degrees * .pi / 180
            return Sighting(classIndex: cls(name), confidence: confidence, direction: SIMD3(sin(a), 0, -cos(a)), radius: radius)
        }
        var verdicts = memory.observe([sighting("wardrobe", 0.5, at: 0), sighting("bathtub", 0.6, at: 60)], eye: eye)
        XCTAssertEqual(verdicts.map(\.sightings), [1, 1])
        XCTAssertFalse(spec.shows(sightings: 1, tracked: 0), "seen once: not outlined yet")
        // The wardrobe again, called a shelf this time, a little aside (the phone turned and took a step).
        // The bathtub was the detector's invention for one frame, and never shows.
        verdicts = memory.observe([sighting("shelf", 0.45, at: 4)], eye: eye + SIMD3(0.1, 0, 0))
        XCTAssertEqual(verdicts[0].sightings, 2)
        XCTAssertEqual(spec.info(verdicts[0].classIndex)?.label, "wardrobe", "the most voted name, not this frame's")
        XCTAssertTrue(spec.shows(sightings: 2, tracked: 0))
        // Something small of the same kin somewhere else is another thing.
        verdicts = memory.observe([sighting("cabinet", 0.5, at: 90, radius: 0.05)], eye: eye)
        XCTAssertEqual(verdicts[0].sightings, 1)
        // A piece the room map already knows is outlined at once.
        XCTAssertTrue(spec.shows(sightings: 1, tracked: 5))
        // What has not been seen for a while is forgotten.
        for _ in 0...spec.screen.forgetAnalyses { _ = memory.observe([], eye: eye) }
        verdicts = memory.observe([sighting("wardrobe", 0.5, at: 0)], eye: eye)
        XCTAssertEqual(verdicts[0].sightings, 1)
    }

    func testADirectionIsWhereAPixelLooks() {
        // A camera turned a quarter turn to the left about the vertical: its centre looks along -x.
        var transform = simd_float4x4(simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0)))
        transform.columns.3 = SIMD4(1, 1.4, 2, 1)
        let camera = PinholeCamera(fx: 1000, fy: 1000, cx: 960, cy: 720, width: 1920, height: 1440, transform: transform)
        let d = camera.direction(u: 960, v: 720)
        XCTAssertEqual(d.x, -1, accuracy: 1e-4)
        XCTAssertEqual(camera.position, SIMD3(1, 1.4, 2))
        // The same thing, in upright image coordinates, through the phone's landscape sensor.
        let s = Sighting(classIndex: 0, confidence: 1, centre: SIMD2(0.5, 0.5), size: SIMD2(0.2, 0.2), camera: camera, sensorLandscape: true)
        XCTAssertEqual(s.direction.x, -1, accuracy: 1e-4)
        XCTAssertEqual(s.radius, atan(sqrt(288 * 288 + 384 * 384) / 2 / 1000), accuracy: 1e-4)
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
