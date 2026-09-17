// swift-tools-version:5.9
// Capture rules: platform-agnostic guidance for recording a room.
// No ARKit, no UIKit: the iPhone app feeds FrameSamples from ARKit, and an
// Android port can feed the same samples from ARCore and reuse
// capture-rules.json (thresholds and messages).
import PackageDescription

let package = Package(
    name: "CaptureRules",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "CaptureRules", targets: ["CaptureRules"])],
    targets: [
        .target(name: "CaptureRules", resources: [.process("Resources")]),
        .testTarget(name: "CaptureRulesTests", dependencies: ["CaptureRules"]),
    ]
)
