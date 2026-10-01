import Foundation
import simd

/// Metres from a MoGe point map.
///
/// MoGe predicts, for every pixel, a camera-space point (x, y, z) whose z is
/// right up to one unknown offset, plus a scale that turns the lot into
/// metres. With the camera's focal length known, the offset follows from the
/// projection itself: focal * x / (z + shift) must land on the pixel's own
/// view-plane coordinate u, so every pixel away from the centre gives an
/// estimate of the shift, and the median of them is robust to the few that
/// are wrong. Conventions are MoGe's: the optical centre is the image centre,
/// u and v span +-aspect / sqrt(1 + aspect^2) and +-1 / sqrt(1 + aspect^2),
/// and the focal is relative to half the image diagonal.
public enum MetricDepth {
    /// Pixels nearer the centre than this (in view-plane units) say little about the shift.
    public static let minimumOffset: Float = 0.15

    /// The focal length relative to half the diagonal, from a focal length in
    /// pixels of a `width` x `height` image.
    public static func relativeFocal(pixels: Float, width: Float, height: Float) -> Float {
        pixels / ((width * width + height * height).squareRoot() / 2)
    }

    /// The view-plane coordinate of column `x` (0-based) in a map `width` wide.
    public static func u(_ x: Int, width: Int, height: Int) -> Float {
        let aspect = Float(width) / Float(height)
        let span = aspect / (1 + aspect * aspect).squareRoot()
        return Float(width) <= 1 ? 0 : -span * Float(width - 1) / Float(width) + Float(x) * (2 * span * Float(width - 1) / Float(width)) / Float(width - 1)
    }

    public static func v(_ y: Int, width: Int, height: Int) -> Float {
        let aspect = Float(width) / Float(height)
        let span = 1 / (1 + aspect * aspect).squareRoot()
        return Float(height) <= 1 ? 0 : -span * Float(height - 1) / Float(height) + Float(y) * (2 * span * Float(height - 1) / Float(height)) / Float(height - 1)
    }

    /// The z shift that makes the point map project with `focal`. `points` is
    /// row-major (y, x) with 3 floats a pixel; `mask` one float a pixel, valid
    /// above 0.5. Samples every `step` pixels. Nil when too few pixels can say.
    public static func shift(points: UnsafePointer<Float>, mask: UnsafePointer<Float>, width: Int, height: Int,
                             focal: Float, step: Int = 4) -> Float? {
        var estimates: [Float] = []
        estimates.reserveCapacity((width / step) * (height / step) * 2)
        var y = step / 2
        while y < height {
            let vv = v(y, width: width, height: height)
            var x = step / 2
            while x < width {
                let i = y * width + x
                if mask[i] > 0.5 {
                    let px = points[i * 3], py = points[i * 3 + 1], pz = points[i * 3 + 2]
                    let uu = u(x, width: width, height: height)
                    if abs(uu) > minimumOffset { estimates.append(focal * px / uu - pz) }
                    if abs(vv) > minimumOffset { estimates.append(focal * py / vv - pz) }
                }
                x += step
            }
            y += step
        }
        guard estimates.count >= 50 else { return nil }
        estimates.sort()
        return estimates[estimates.count / 2]
    }

    /// Depth in metres for every pixel (nan where the mask says there is none):
    /// (z + shift) * scale.
    public static func metres(points: UnsafePointer<Float>, mask: UnsafePointer<Float>, width: Int, height: Int,
                              shift: Float, scale: Float, into depth: inout [Float]) {
        if depth.count != width * height { depth = [Float](repeating: .nan, count: width * height) }
        for i in 0..<(width * height) {
            depth[i] = mask[i] > 0.5 ? (points[i * 3 + 2] + shift) * scale : .nan
        }
    }
}
