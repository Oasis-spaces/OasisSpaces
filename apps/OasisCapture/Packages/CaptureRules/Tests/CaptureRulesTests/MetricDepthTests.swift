import XCTest
@testable import CaptureRules

final class MetricDepthTests: XCTestCase {
    /// A synthetic room: a wall 3 m away and a floor, seen by a camera of known
    /// focal length; the point map is built from the true depth, then its z is
    /// shifted and everything is divided by a scale. Recovery must undo both.
    func testShiftAndScaleAreRecoveredFromAProjectedPointMap() {
        let width = 64, height = 48
        let focalPixels: Float = 60, scale: Float = 0.5, hiddenShift: Float = -1.7
        let focal = MetricDepth.relativeFocal(pixels: focalPixels, width: Float(width), height: Float(height))
        // u = focal * x / z, so x = u * z / focal (and y likewise).
        var points = [Float](repeating: 0, count: width * height * 3)
        var mask = [Float](repeating: 1, count: width * height)
        var truth = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let u = MetricDepth.u(x, width: width, height: height), v = MetricDepth.v(y, width: width, height: height)
                // Depth: a floor in the lower half (nearer at the bottom), a wall at 3 m above.
                let z: Float = v > 0.1 ? 3 / (1 + 4 * (v - 0.1)) : 3
                truth[y * width + x] = z
                let i = (y * width + x) * 3
                points[i] = (u * z / focal) / scale
                points[i + 1] = (v * z / focal) / scale
                points[i + 2] = z / scale + hiddenShift
            }
        }
        mask[0] = 0; mask[1] = 0   // a couple of pixels the model refused
        let shift = points.withUnsafeBufferPointer { p in mask.withUnsafeBufferPointer { m in
            MetricDepth.shift(points: p.baseAddress!, mask: m.baseAddress!, width: width, height: height, focal: focal, step: 2) } }
        XCTAssertNotNil(shift)
        XCTAssertEqual(shift!, -hiddenShift, accuracy: 1e-3)
        var depth: [Float] = []
        points.withUnsafeBufferPointer { p in mask.withUnsafeBufferPointer { m in
            MetricDepth.metres(points: p.baseAddress!, mask: m.baseAddress!, width: width, height: height, shift: shift!, scale: scale, into: &depth) } }
        XCTAssertTrue(depth[0].isNaN && depth[1].isNaN)
        for i in 2..<depth.count { XCTAssertEqual(depth[i], truth[i], accuracy: 1e-3) }
        // The view-plane grid matches MoGe's: corners at +-aspect/diag, +-1/diag scaled by (n-1)/n.
        let aspect = Float(width) / Float(height)
        XCTAssertEqual(MetricDepth.u(0, width: width, height: height), -aspect / (1 + aspect * aspect).squareRoot() * Float(width - 1) / Float(width), accuracy: 1e-6)
        XCTAssertEqual(MetricDepth.v(height - 1, width: width, height: height), 1 / (1 + aspect * aspect).squareRoot() * Float(height - 1) / Float(height), accuracy: 1e-6)
        XCTAssertEqual(MetricDepth.u(width / 2, width: width, height: height), MetricDepth.u(0, width: width, height: height) + 2 * aspect / (1 + aspect * aspect).squareRoot() * Float(width - 1) / Float(width) * Float(width / 2) / Float(width - 1), accuracy: 1e-5)
    }

    func testTooFewPixelsGiveNoShift() {
        let points = [Float](repeating: 0, count: 8 * 8 * 3), mask = [Float](repeating: 1, count: 64)
        let shift = points.withUnsafeBufferPointer { p in mask.withUnsafeBufferPointer { m in
            MetricDepth.shift(points: p.baseAddress!, mask: m.baseAddress!, width: 8, height: 8, focal: 1, step: 4) } }
        XCTAssertNil(shift)
    }
}
