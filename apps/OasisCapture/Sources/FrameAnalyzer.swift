import ARKit
import CaptureRules

/// Turns an ARFrame into a FrameSample for the rules, and samples colours for
/// the live scan. Everything here runs on the capture queue and must stay
/// cheap: ARKit delivers 30-60 frames a second and stalls if a frame is held.
final class FrameAnalyzer {
    private var frameIndex = 0

    func sample(_ frame: ARFrame) -> FrameSample {
        frameIndex += 1
        let camera = frame.camera
        let transform = camera.transform
        let position = SIMD3<Float>(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
        // The camera looks along its -Z axis, whichever way the phone is held.
        let forward = -SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)

        let luma = lumaStats(frame.capturedImage)
        let features = frame.rawFeaturePoints?.points ?? []
        // People come from the room segmentation (CaptureController).
        let people: Int? = nil

        return FrameSample(
            time: frame.timestamp,
            position: position,
            forward: simd_normalize(forward),
            tracking: tracking(camera.trackingState),
            exposureDuration: camera.exposureDuration,
            focalPixels: Double(camera.intrinsics.columns.0.x),
            ambientIntensity: frame.lightEstimate.map { Double($0.ambientIntensity) },
            meanLuma: luma.mean,
            saturatedShare: luma.saturated,
            featurePoints: features.count,
            nearestSurface: nearestSurface(features, camera: camera),
            peopleInView: people)
    }

    private func tracking(_ state: ARCamera.TrackingState) -> FrameSample.Tracking {
        switch state {
        case .normal: return .normal
        case .notAvailable: return .notAvailable
        case .limited(let reason):
            switch reason {
            case .initializing: return .initializing
            case .excessiveMotion: return .excessiveMotion
            case .insufficientFeatures: return .insufficientFeatures
            case .relocalizing: return .relocalizing
            @unknown default: return .initializing
            }
        @unknown default: return .notAvailable
        }
    }

    /// Mean brightness and the share of blown-out pixels, from a sparse grid
    /// over the luma plane (full-range 8-bit Y).
    private func lumaStats(_ buffer: CVPixelBuffer) -> (mean: Double, saturated: Double) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return (0.5, 0) }
        let width = CVPixelBufferGetWidthOfPlane(buffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        let step = max(8, width / 64)
        var sum = 0, count = 0, blown = 0
        var y = step / 2
        while y < height {
            var x = step / 2
            while x < width {
                let v = Int(pixels[y * stride + x])
                sum += v
                if v >= 250 { blown += 1 }
                count += 1
                x += step
            }
            y += step
        }
        guard count > 0 else { return (0.5, 0) }
        return (Double(sum) / Double(count) / 255, Double(blown) / Double(count))
    }

    /// How close the nearest surfaces in view are: the 10th percentile depth of
    /// tracked feature points in front of the camera.
    private func nearestSurface(_ points: [SIMD3<Float>], camera: ARCamera) -> Double? {
        guard points.count >= 20 else { return nil }
        let toCamera = camera.transform.inverse
        var depths: [Float] = []
        depths.reserveCapacity(points.count)
        for p in points {
            let c = toCamera * SIMD4<Float>(p, 1)
            if c.z < -0.05 { depths.append(-c.z) }   // in front: the camera looks down -Z
        }
        guard depths.count >= 20 else { return nil }
        depths.sort()
        return Double(depths[depths.count / 10])
    }

}

/// The colour of a world point as the camera sees it this frame, or nil when
/// it falls outside the image.
func sampleColor(of point: SIMD3<Float>, frame: ARFrame) -> SIMD3<Float>? {
    let buffer = frame.capturedImage
    let size = frame.camera.imageResolution
    let projected = frame.camera.projectPoint(point, orientation: .landscapeRight, viewportSize: size)
    let x = Int(projected.x), y = Int(projected.y)
    guard x >= 0, y >= 0, x < Int(size.width), y < Int(size.height) else { return nil }

    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let yBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
          let cBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else { return nil }
    let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
    let cStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
    let luma = Float(yBase.assumingMemoryBound(to: UInt8.self)[y * yStride + x])
    let c = cBase.assumingMemoryBound(to: UInt8.self)
    let ci = (y / 2) * cStride + (x / 2) * 2
    let cb = Float(c[ci]) - 128, cr = Float(c[ci + 1]) - 128
    let r = luma + 1.402 * cr
    let g = luma - 0.344136 * cb - 0.714136 * cr
    let b = luma + 1.772 * cb
    return SIMD3<Float>(r, g, b).clamped(lowerBound: .zero, upperBound: SIMD3<Float>(repeating: 255)) / 255
}
