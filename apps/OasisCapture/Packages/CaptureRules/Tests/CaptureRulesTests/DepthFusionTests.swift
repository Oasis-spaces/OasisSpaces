import XCTest
import simd
@testable import CaptureRules

final class DepthFusionTests: XCTestCase {
    func testFitRecoversScaleAndOffsetDespiteOutliers() {
        // The model says d = 3 / z + 0.2 (up to noise); a few tracked points lie (a mirror).
        var predicted: [Float] = [], metres: [Float] = []
        var seed: UInt64 = 7
        func noise() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) * 0.02 - 0.01 }
        for i in 0..<40 {
            let z = 0.8 + Float(i) * 0.15
            metres.append(z)
            predicted.append(3 / z + 0.2 + noise())
        }
        metres += [2.0, 3.0, 1.5]
        predicted += [0.9, 2.8, 1.1]   // nonsense
        let fit = DepthScale.fit(predicted: predicted, metres: metres)!
        XCTAssertEqual(fit.metres(3 / 2.5 + 0.2)!, 2.5, accuracy: 0.05)
        XCTAssertEqual(fit.metres(3 / 5.0 + 0.2)!, 5.0, accuracy: 0.15)
        XCTAssertLessThan(fit.error, 0.02)
        XCTAssertNil(DepthScale.fit(predicted: [1, 2, 3], metres: [1, 2, 3]), "too few points")
        XCTAssertNil(DepthScale.fit(predicted: metres, metres: metres, minimum: 12), "closer must give a larger value")
    }

    func testBackProjectionRoundTrips() {
        // A camera at (1, 1.4, 2) turned 30 degrees, looking slightly down.
        let yaw = simd_quatf(angle: .pi / 6, axis: SIMD3(0, 1, 0))
        let pitch = simd_quatf(angle: -0.2, axis: SIMD3(1, 0, 0))
        var transform = simd_float4x4(yaw * pitch)
        transform.columns.3 = SIMD4(1, 1.4, 2, 1)
        let camera = PinholeCamera(fx: 1500, fy: 1500, cx: 960, cy: 720, width: 1920, height: 1440, transform: transform)
        for (u, v, z) in [(Float(960), Float(720), Float(2)), (100, 50, 3.5), (1800, 1400, 0.8)] {
            let world = camera.worldPoint(u: u, v: v, depth: z)
            let back = camera.project(world)!
            XCTAssertEqual(back.u, u, accuracy: 0.01)
            XCTAssertEqual(back.v, v, accuracy: 0.01)
            XCTAssertEqual(back.depth, z, accuracy: 0.001)
        }
        // The centre pixel at depth 1 is one metre along the camera's -z.
        let ahead = camera.worldPoint(u: 960, v: 720, depth: 1)
        let forward = -SIMD3(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        XCTAssertEqual(simd_distance(ahead, SIMD3(1, 1.4, 2) + forward), 0, accuracy: 1e-4)
        XCTAssertNil(camera.project(SIMD3(1, 1.4, 2) - forward * 2), "behind the camera")
    }
}
