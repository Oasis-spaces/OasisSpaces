import XCTest
import simd
@testable import CaptureRules

final class InstanceDecoderTests: XCTestCase {
    let spec = ObjectSpec.bundled()

    func testSpecHasNoClothesAndKnowsItsKin() {
        XCTAssertNil(spec.index(of: "apparel"))
        XCTAssertNil(spec.index(of: "clothes"))
        XCTAssertEqual(spec.classes.count, 71)
        XCTAssertEqual(spec.kinGroup(spec.index(of: "bed")!), "sitting")
        XCTAssertEqual(spec.kinGroup(spec.index(of: "armchair")!), "sitting")
        XCTAssertEqual(spec.kinGroup(spec.index(of: "wardrobe")!), spec.kinGroup(spec.index(of: "desk")!))
        XCTAssertEqual(spec.kinGroup(spec.index(of: "lamp")!), "lamp")
        XCTAssertFalse(spec.info(spec.index(of: "painting")!)!.boxed)
        XCTAssertTrue(spec.info(spec.index(of: "sofa")!)!.boxed)
        XCTAssertEqual(spec.inputWidth, 480)
        XCTAssertEqual(spec.inputHeight, 640)
    }

    /// Builds raw outputs: `anchors` detections (box in input pixels, class,
    /// score, prototype index whose mask they use) and prototypes that are +6
    /// inside a rectangle and -6 outside.
    private func raw(_ detections: [(box: (Float, Float, Float, Float), cls: String, score: Float, proto: Int)],
                     protoRects: [(Int, Int, Int, Int)], mw: Int = 24, mh: Int = 32) -> ([Float], [Float]) {
        let nc = spec.classes.count
        let anchors = detections.count
        var predictions = [Float](repeating: 0, count: (4 + nc + 32) * anchors)
        for (a, d) in detections.enumerated() {
            predictions[0 * anchors + a] = d.box.0; predictions[1 * anchors + a] = d.box.1
            predictions[2 * anchors + a] = d.box.2; predictions[3 * anchors + a] = d.box.3
            predictions[(4 + spec.index(of: d.cls)!) * anchors + a] = d.score
            predictions[(4 + nc + d.proto) * anchors + a] = 1
        }
        var protos = [Float](repeating: -6, count: 32 * mw * mh)
        for (k, r) in protoRects.enumerated() {
            for y in r.1..<r.3 { for x in r.0..<r.2 { protos[k * mw * mh + y * mw + x] = 6 } }
        }
        return (predictions, protos)
    }

    func testDecodesBoxesMasksAndSuppressesLookAlikes() {
        // Input 480x640; masks 24x32 (1 mask px = 20 input px).
        let (p, q) = raw([
            (box: (240, 320, 240, 320), cls: "bed", score: 0.9, proto: 0),       // centre, 240x320 px
            (box: (240, 320, 240, 320), cls: "sofa", score: 0.6, proto: 0),      // the same thing again: suppressed (same kin)
            (box: (200, 300, 100, 60), cls: "pillow", score: 0.5, proto: 1),     // on the bed: different kin, kept
            (box: (100, 100, 100, 100), cls: "lamp", score: 0.1, proto: 2),      // below the threshold
            (box: (400, 100, 60, 60), cls: "plant", score: 0.4, proto: 3),       // mask empty (proto never positive): dropped
        ], protoRects: [(6, 8, 18, 24), (8, 14, 12, 16), (0, 0, 0, 0), (0, 0, 0, 0)])
        let out = InstanceDecoder.decode(predictions: p, anchors: 5, protos: q, maskWidth: 24, maskHeight: 32, spec: spec)
        XCTAssertEqual(out.map { spec.info($0.classIndex)!.label }, ["bed", "pillow"], "\(out.map(\.classIndex))")
        let bed = out[0]
        XCTAssertEqual(bed.confidence, 0.9)
        XCTAssertEqual(bed.minX, 0.25, accuracy: 1e-5); XCTAssertEqual(bed.maxX, 0.75, accuracy: 1e-5)
        XCTAssertEqual(bed.minY, 0.25, accuracy: 1e-5); XCTAssertEqual(bed.maxY, 0.75, accuracy: 1e-5)
        XCTAssertEqual(bed.area, 12 * 16, "the prototype rectangle, which lies inside the box")
        XCTAssertTrue(bed.inside(x: 10, y: 12))
        XCTAssertFalse(bed.inside(x: 2, y: 2))
        let pillow = out[1]
        XCTAssertEqual(pillow.area, 4 * 2)
        // Outline of the bed's mask: a rectangle, four corners.
        let polygon = MaskOutline.polygon(of: bed.mask, width: 24, height: 32)
        XCTAssertEqual(polygon.count, 4, "\(polygon)")
        XCTAssertEqual(polygon.map(\.x).min()!, 6.5 / 24, accuracy: 1e-9)
        XCTAssertEqual(polygon.map(\.y).max()!, 23.5 / 32, accuracy: 1e-9)
        let c = MaskOutline.centroid(of: bed.mask, width: 24, height: 32)
        XCTAssertEqual(c.x, 12.0 / 24, accuracy: 1e-9)
    }

    func testMaskIsCutToItsBoxAndToItsLargestPiece() {
        // The prototype lights two separate blobs; the box covers both, the bigger one wins.
        let (p, q) = raw([(box: (240, 320, 480, 640), cls: "sofa", score: 0.8, proto: 0)],
                         protoRects: [(2, 2, 10, 10)])
        var protos = q
        for y in 20..<24 { for x in 14..<18 { protos[y * 24 + x] = 6 } }   // a second, smaller blob
        let out = InstanceDecoder.decode(predictions: p, anchors: 1, protos: protos, maskWidth: 24, maskHeight: 32, spec: spec)
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].area, 64)
        XCTAssertFalse(out[0].inside(x: 15, y: 21))
        // A box that only covers part of the blob cuts the mask to the box.
        let (p2, _) = raw([(box: (60, 60, 40, 40), cls: "sofa", score: 0.8, proto: 0)], protoRects: [(2, 2, 10, 10)])
        let cut = InstanceDecoder.decode(predictions: p2, anchors: 1, protos: q, maskWidth: 24, maskHeight: 32, spec: spec)
        XCTAssertEqual(cut.count, 1)
        XCTAssertLessThan(cut[0].area, 64)
        XCTAssertGreaterThan(cut[0].area, 0)
    }
}
