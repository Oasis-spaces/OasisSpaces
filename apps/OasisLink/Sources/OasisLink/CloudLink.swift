import Foundation

/// Where the cloud relay is (Resources/cloud.json).
public struct CloudConfig: Codable, Sendable {
    public var url: URL

    public init(url: URL) { self.url = url }

    public static func load() -> CloudConfig? {
        guard let url = Bundle.module.url(forResource: "cloud", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(CloudConfig.self, from: data)
    }
}

/// The cloud relay as a station: the same client as for a Mac on the local
/// network, with the account's session as the token. Nothing to pair: being
/// signed in is the pairing.
@MainActor
public final class CloudLink {
    public let config: CloudConfig?
    public let account: AccountStore
    /// Files are uploaded in parts of this many bytes (under the storage's object limit).
    public static let partBytes = 40 * 1024 * 1024

    public init(account: AccountStore, config: CloudConfig? = CloudConfig.load()) {
        self.account = account
        self.config = config
    }

    /// Whether the cloud can be used right now: configured and signed in.
    public var isAvailable: Bool { config != nil && account.isSignedIn }

    /// A client whose token is a session valid for the next minute, or nil when signed out.
    public func client() async -> StationClient? {
        guard let config, let session = await account.validSession() else { return nil }
        return StationClient(baseURL: config.url, token: session.accessToken)
    }
}
