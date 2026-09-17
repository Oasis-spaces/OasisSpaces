import simd

/// What the phone knows about one camera frame. Everything is optional except
/// time, pose and tracking, so a phone without a sensor (no light estimate, no
/// people detector this frame) still gets the rules it can.
///
/// Coordinates are gravity-aligned and in metres, y pointing up (ARKit and
/// ARCore world space both are).
public struct FrameSample: Sendable {
    public enum Tracking: Sendable, Equatable {
        case normal
        case initializing
        case excessiveMotion
        case insufficientFeatures
        case relocalizing
        case notAvailable
    }

    public var time: Double
    public var position: SIMD3<Float>
    /// The direction the camera looks, a unit vector in world space.
    public var forward: SIMD3<Float>
    public var tracking: Tracking
    /// Exposure time of the frame in seconds.
    public var exposureDuration: Double?
    /// Focal length in image pixels (fx).
    public var focalPixels: Double?
    /// Ambient light in lumens (ARKit: about 1000 in a well-lit room).
    public var ambientIntensity: Double?
    /// Mean brightness 0...1 and the share of near-white pixels.
    public var meanLuma: Double?
    public var saturatedShare: Double?
    public var featurePoints: Int?
    /// Distance to the nearest surfaces in view (a low percentile of feature depths).
    public var nearestSurface: Double?
    /// People detected in the frame; nil when detection did not run this frame.
    public var peopleInView: Int?

    public init(time: Double, position: SIMD3<Float>, forward: SIMD3<Float>, tracking: Tracking,
                exposureDuration: Double? = nil, focalPixels: Double? = nil,
                ambientIntensity: Double? = nil, meanLuma: Double? = nil, saturatedShare: Double? = nil,
                featurePoints: Int? = nil, nearestSurface: Double? = nil, peopleInView: Int? = nil) {
        self.time = time
        self.position = position
        self.forward = forward
        self.tracking = tracking
        self.exposureDuration = exposureDuration
        self.focalPixels = focalPixels
        self.ambientIntensity = ambientIntensity
        self.meanLuma = meanLuma
        self.saturatedShare = saturatedShare
        self.featurePoints = featurePoints
        self.nearestSurface = nearestSurface
        self.peopleInView = peopleInView
    }

    /// Degrees above (+) or below (-) the horizon.
    public var pitch: Double {
        Double(asin(max(-1, min(1, forward.y)))) * 180 / .pi
    }

    /// Compass heading of the view in degrees, 0...360.
    public var heading: Double {
        let h = Double(atan2(forward.x, -forward.z)) * 180 / .pi
        return h < 0 ? h + 360 : h
    }
}
