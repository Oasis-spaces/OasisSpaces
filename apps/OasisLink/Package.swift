// swift-tools-version:5.9
// OasisLink: how the phone app and the Mac app talk over the local network.
// Shared models, a small HTTP server (Mac), a client (phone) and Bonjour
// discovery. The phone finds the Mac, sends a recording, starts the
// analysis, follows its progress and fetches the results.
import PackageDescription

let package = Package(
    name: "OasisLink",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "OasisLink", targets: ["OasisLink"])],
    targets: [
        .target(name: "OasisLink", resources: [.process("Resources")]),
        .testTarget(name: "OasisLinkTests", dependencies: ["OasisLink"]),
    ]
)
