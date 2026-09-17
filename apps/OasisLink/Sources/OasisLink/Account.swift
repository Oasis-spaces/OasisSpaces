import CryptoKit
import Foundation

/// Where accounts live: a Supabase project's URL and its public (anon /
/// publishable) key, from supabase.json. The key is safe to ship in an app.
public struct AccountConfig: Codable, Sendable, Equatable {
    public var url: URL
    public var anonKey: String

    public init(url: URL, anonKey: String) {
        self.url = url
        self.anonKey = anonKey
    }

    /// supabase.json in the app's bundle, else the package's own (an empty
    /// template until a project is set up); nil while it is empty.
    public static func load(bundle: Bundle = .main) -> AccountConfig? {
        guard let file = bundle.url(forResource: "supabase", withExtension: "json")
                ?? Bundle.module.url(forResource: "supabase", withExtension: "json"),
              let data = try? Data(contentsOf: file),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: String],
              let text = raw["url"], !text.isEmpty, let url = URL(string: text),
              let key = raw["anonKey"], !key.isEmpty else { return nil }
        return AccountConfig(url: url, anonKey: key)
    }
}

public struct AccountSession: Codable, Sendable, Equatable {
    public var accessToken: String
    public var refreshToken: String
    public var expiresAt: Date
    public var userID: String
    public var email: String

    public var isExpiring: Bool { expiresAt.timeIntervalSinceNow < 60 }

    /// What a Mac advertises on the network in place of its account id.
    public var ownerTag: String { AccountSession.tag(userID) }

    public static func tag(_ userID: String) -> String {
        SHA256.hash(data: Data("oasis-owner:\(userID)".utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
    }
}

public struct AccountError: Error, LocalizedError, Sendable {
    public var message: String
    public var errorDescription: String? { message }
}

/// Supabase Auth (GoTrue) over its REST API: sign up, sign in with email and
/// password, refresh, and look up who a token belongs to.
public final class AccountClient: @unchecked Sendable {
    public let config: AccountConfig
    private let session: URLSession

    public init(config: AccountConfig) {
        self.config = config
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 20
        session = URLSession(configuration: configuration)
    }

    /// Returns a session, or nil when the project asks new users to confirm
    /// their email first.
    public func signUp(email: String, password: String) async throws -> AccountSession? {
        let json = try await post("auth/v1/signup", ["email": email, "password": password])
        return json["access_token"] != nil ? try Self.session(from: json) : nil
    }

    public func signIn(email: String, password: String) async throws -> AccountSession {
        try Self.session(from: try await post("auth/v1/token?grant_type=password", ["email": email, "password": password]))
    }

    public func refresh(_ current: AccountSession) async throws -> AccountSession {
        try Self.session(from: try await post("auth/v1/token?grant_type=refresh_token", ["refresh_token": current.refreshToken]))
    }

    /// The user id a token belongs to, or nil when it is not valid.
    public func userID(for accessToken: String) async -> String? {
        var request = URLRequest(url: endpoint("auth/v1/user"))
        request.setValue(config.anonKey, forHTTPHeaderField: "apikey")
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        guard let (data, response) = try? await session.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return json["id"] as? String
    }

    private func endpoint(_ path: String) -> URL {
        URL(string: path, relativeTo: config.url.appendingPathComponent("/"))!.absoluteURL
    }

    private func post(_ path: String, _ body: [String: String]) async throws -> [String: Any] {
        var request = URLRequest(url: endpoint(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(config.anonKey, forHTTPHeaderField: "apikey")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await session.data(for: request)
        let json = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let message = (json["msg"] ?? json["error_description"] ?? json["message"] ?? json["error"]) as? String
            throw AccountError(message: message ?? "Sign-in failed")
        }
        return json
    }

    static func session(from json: [String: Any]) throws -> AccountSession {
        guard let access = json["access_token"] as? String, let refresh = json["refresh_token"] as? String,
              let user = json["user"] as? [String: Any], let id = user["id"] as? String else {
            throw AccountError(message: "Unexpected answer from the account server")
        }
        let expiresIn = (json["expires_in"] as? Double) ?? 3600
        return AccountSession(accessToken: access, refreshToken: refresh, expiresAt: Date().addingTimeInterval(expiresIn),
                              userID: id, email: user["email"] as? String ?? "")
    }
}

public struct AccountPairRequest: Codable, Sendable {
    public var accessToken: String
    public var device: String
    public init(accessToken: String, device: String) {
        self.accessToken = accessToken
        self.device = device
    }
}
