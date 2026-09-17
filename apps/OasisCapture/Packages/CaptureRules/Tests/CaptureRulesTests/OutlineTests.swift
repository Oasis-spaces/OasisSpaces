import XCTest
import simd
@testable import CaptureRules

final class OutlineTests: XCTestCase {
    let spec = DetectionSpec.bundled()

    private func id(_ name: String) -> Int32 {
        Int32(spec.classes.first { $0.name == name }!.id)
    }

    /// A 64x64 map filled with one class.
    private func map(filledWith name: String, size: Int = 64) -> [Int32] {
        [Int32](repeating: id(name), count: size * size)
    }

    private func paint(_ m: inout [Int32], _ name: String, x: Range<Int>, y: Range<Int>, size: Int = 64) {
        for yy in y { for xx in x { m[yy * size + xx] = id(name) } }
    }

    func testSpecExcludesClothesAndOutlinesRoomThings() {
        let byName = Dictionary(uniqueKeysWithValues: spec.classes.map { ($0.name, $0) })
        XCTAssertEqual(byName["apparel"]?.outline, false)
        for name in ["wall", "floor", "bed", "table", "cabinet", "wardrobe", "sofa", "rug", "curtain",
                     "chest of drawers", "shelf", "door", "windowpane", "lamp", "refrigerator"] {
            XCTAssertEqual(byName[name]?.outline, true, "\(name) should be outlined")
        }
        XCTAssertEqual(byName["person"]?.outline, false)
    }

    func testRectangleOutlineIsItsFourCorners() {
        var m = map(filledWith: "wall")
        paint(&m, "bed", x: 10..<40, y: 20..<50)
        let result = OutlineExtractor.extract(classes: m, width: 64, height: 64, spec: spec)
        let bed = result.regions.first { $0.label == "bed" }!
        XCTAssertEqual(bed.share, 900.0 / 4096, accuracy: 1e-9)
        XCTAssertEqual(bed.outline.count, 4, "a rectangle simplifies to 4 corners: \(bed.outline)")
        let xs = bed.outline.map(\.x), ys = bed.outline.map(\.y)
        XCTAssertEqual(xs.min()!, 10.5 / 64, accuracy: 1e-9)
        XCTAssertEqual(xs.max()!, 39.5 / 64, accuracy: 1e-9)
        XCTAssertEqual(ys.min()!, 20.5 / 64, accuracy: 1e-9)
        XCTAssertEqual(ys.max()!, 49.5 / 64, accuracy: 1e-9)
        XCTAssertEqual(bed.centroid.x, 25.0 / 64, accuracy: 1e-9)
        // The wall around it is one region too.
        XCTAssertTrue(result.regions.contains { $0.label == "wall" })
    }

    func testLShapeKeepsItsCorners() {
        var m = map(filledWith: "floor")
        paint(&m, "wardrobe", x: 5..<15, y: 5..<40)
        paint(&m, "wardrobe", x: 15..<35, y: 30..<40)
        let result = OutlineExtractor.extract(classes: m, width: 64, height: 64, spec: spec)
        let wardrobe = result.regions.first { $0.label == "wardrobe" }!
        XCTAssertEqual(wardrobe.outline.count, 6, "an L has 6 corners: \(wardrobe.outline)")
    }

    func testClothesTinyBitsAndPeople() {
        var m = map(filledWith: "wall")
        paint(&m, "apparel", x: 0..<20, y: 0..<20)      // clothes: never outlined
        paint(&m, "lamp", x: 50..<52, y: 50..<52)        // 4 px, under minRegionShare
        paint(&m, "person", x: 30..<40, y: 0..<20)
        paint(&m, "curtain", x: 0..<8, y: 40..<64)
        paint(&m, "curtain", x: 56..<64, y: 40..<64)     // a second curtain region
        let result = OutlineExtractor.extract(classes: m, width: 64, height: 64, spec: spec)
        let labels = result.regions.map(\.label)
        XCTAssertFalse(labels.contains("apparel"))
        XCTAssertFalse(labels.contains("lamp"))
        XCTAssertFalse(labels.contains("person"))
        XCTAssertEqual(labels.filter { $0 == "curtain" }.count, 2)
        XCTAssertEqual(result.personShare, 200.0 / 4096, accuracy: 1e-9)
    }

    func testKinClassesFormOneRegionWithTheMajorityLabel() {
        var m = map(filledWith: "wall")
        paint(&m, "bed", x: 10..<30, y: 20..<50)
        paint(&m, "sofa", x: 30..<40, y: 20..<50)
        let result = OutlineExtractor.extract(classes: m, width: 64, height: 64, spec: spec)
        let sitting = result.regions.filter { ["bed", "sofa"].contains($0.label) }
        XCTAssertEqual(sitting.count, 1, "\(result.regions.map(\.label))")
        XCTAssertEqual(sitting[0].label, "bed")
        XCTAssertEqual(sitting[0].share, 900.0 / 4096, accuracy: 1e-9)
    }

    func testSinglePixelAndDiagonalShapesDoNotHang() {
        var m = map(filledWith: "wall", size: 16)
        for i in 0..<16 { m[i * 16 + i] = id("rug") }      // a 1-px diagonal line
        let spec2 = { var s = spec; s.minRegionShare = 0; return s }()
        let result = OutlineExtractor.extract(classes: m, width: 16, height: 16, spec: spec2)
        XCTAssertFalse(result.regions.isEmpty)
    }
}
