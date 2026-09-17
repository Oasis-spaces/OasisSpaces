import Foundation
import simd

/// Turns a depth model's relative output into metres, and pixels into world points.
///
/// Monocular depth models (Depth Anything) return relative inverse depth: a
/// value per pixel that is larger the closer the surface is, up to an unknown
/// scale and offset. The phone's tracking gives a few dozen points per frame
/// whose metric depth is known, so a line 1/z = a * d + b is fitted through
/// those and applied to every pixel.
public enum DepthScale {
    public struct Fit: Sendable, Equatable {
        public var a: Float
        public var b: Float
        public var samples: Int
        /// Median relative error of the fit on the points it was fitted to.
        public var error: Float

        /// Metres for a predicted value.
        public func metres(_ predicted: Float) -> Float? {
            let inverse = a * predicted + b
            return inverse > 1e-4 ? 1 / inverse : nil
        }
    }

    /// `predicted` and `metres` are paired: the model's value at a pixel and the
    /// true depth there. Least squares on inverse depth, repeated with the
    /// worst-fitting third dropped once (a tracked point on a window or a mirror
    /// has no honest depth). Needs at least `minimum` pairs.
    public static func fit(predicted: [Float], metres: [Float], minimum: Int = 12) -> Fit? {
        precondition(predicted.count == metres.count)
        var pairs = zip(predicted, metres).filter { $0.1 > 0.1 && $0.1 < 20 && $0.0.isFinite }
        guard pairs.count >= minimum else { return nil }
        var fit = solve(pairs)
        // Drop the worst third and fit again.
        let residuals = pairs.map { abs((fit.a * $0.0 + fit.b) - 1 / $0.1) }
        let cutoff = residuals.sorted()[residuals.count * 2 / 3]
        let kept = zip(pairs, residuals).filter { $0.1 <= cutoff }.map(\.0)
        if kept.count >= minimum {
            pairs = kept
            fit = solve(pairs)
        }
        guard fit.a > 0 else { return nil }   // closer must mean a larger value
        let errors = pairs.map { pair -> Float in
            guard let z = fit.metres(pair.0) else { return 1 }
            return abs(z - pair.1) / pair.1
        }.sorted()
        return Fit(a: fit.a, b: fit.b, samples: pairs.count, error: errors[errors.count / 2])
    }

    private static func solve(_ pairs: [(Float, Float)]) -> Fit {
        // y = 1/z against x = predicted.
        var sx: Double = 0, sy: Double = 0, sxx: Double = 0, sxy: Double = 0
        for (x, z) in pairs {
            let y = 1 / Double(z)
            sx += Double(x); sy += y; sxx += Double(x) * Double(x); sxy += Double(x) * y
        }
        let n = Double(pairs.count)
        let denominator = n * sxx - sx * sx
        guard abs(denominator) > 1e-12 else { return Fit(a: 0, b: 0, samples: pairs.count, error: 1) }
        let a = (n * sxy - sx * sy) / denominator
        let b = (sy - a * sx) / n
        return Fit(a: Float(a), b: Float(b), samples: pairs.count, error: 0)
    }
}

/// A pinhole camera: pixels to rays and back, in the image's own (landscape
/// sensor) coordinates, and camera to world through its transform.
public struct PinholeCamera: Sendable {
    /// Focal lengths and principal point in pixels.
    public var fx: Float, fy: Float, cx: Float, cy: Float
    public var width: Int, height: Int
    /// Camera to world. The camera looks along its -z, x right, y up.
    public var transform: simd_float4x4

    public init(fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int, transform: simd_float4x4) {
        self.fx = fx; self.fy = fy; self.cx = cx; self.cy = cy
        self.width = width; self.height = height
        self.transform = transform
    }

    /// The world point at depth `z` metres behind pixel (u, v).
    public func worldPoint(u: Float, v: Float, depth z: Float) -> SIMD3<Float> {
        let x = (u - cx) / fx * z
        let y = -(v - cy) / fy * z    // image v grows downwards, camera y grows upwards
        let p = transform * SIMD4(x, y, -z, 1)
        return SIMD3(p.x, p.y, p.z)
    }

    /// The pixel a world point lands on, and its depth; nil when behind the camera.
    public func project(_ world: SIMD3<Float>) -> (u: Float, v: Float, depth: Float)? {
        let c = transform.inverse * SIMD4(world, 1)
        let z = -c.z
        guard z > 0.01 else { return nil }
        return (fx * c.x / z + cx, -fy * c.y / z + cy, z)
    }
}
