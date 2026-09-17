import Foundation

/// Thresholds and messages, read from capture-rules.json.
public struct RuleConfig: Codable, Sendable {
    public var holdSeconds: Double
    public var releaseSeconds: Double
    public var hapticGapSeconds: Double

    public var maxAngularSpeed: Double
    public var maxLinearSpeed: Double
    public var maxBlurDegrees: Double

    public var spinWindowSeconds: Double
    public var spinMinTurn: Double
    public var spinMaxTravel: Double

    public var floorCheckAfterSeconds: Double
    public var floorPitch: Double

    public var minAmbientIntensity: Double
    public var minMeanLuma: Double
    public var maxSaturatedShare: Double
    public var minSurfaceDistance: Double
    public var minFeaturePoints: Int

    public var coverageSectors: Int
    public var coverageMinPitch: Double
    public var coverageMaxPitch: Double
    public var coverageHintAfterSeconds: Double
    public var coverageGoal: Double

    public var minDurationSeconds: Double
    public var loopCloseDistance: Double

    public var messages: [String: String]
    public var tips: Tips

    /// The bundled capture-rules.json.
    public static func bundled() -> RuleConfig {
        guard let url = Bundle.module.url(forResource: "capture-rules", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let config = try? JSONDecoder().decode(RuleConfig.self, from: data) else {
            fatalError("capture-rules.json is missing or does not match RuleConfig")
        }
        return config
    }

    public func message(_ rule: Rule) -> String {
        messages[rule.rawValue] ?? rule.rawValue
    }
}

/// Advice shown before recording: what a good scan needs, and what makes it
/// the best it can be.
public struct Tip: Codable, Sendable, Identifiable {
    public var icon: String
    public var title: String
    public var detail: String
    public var id: String { title }
}

public struct Tips: Codable, Sendable {
    public var good: [Tip]
    public var best: [Tip]
}
