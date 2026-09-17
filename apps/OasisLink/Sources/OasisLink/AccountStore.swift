import Foundation
import Security

/// Signed-in state shared by the Mac and phone apps: the session in the
/// Keychain, refreshed when it is about to expire.
@MainActor
public final class AccountStore: ObservableObject {
    public let config: AccountConfig?
    @Published public private(set) var session: AccountSession?
    @Published public private(set) var busy = false
    @Published public private(set) var error: String?
    /// Sign-up succeeded but the project wants the email confirmed first.
    @Published public private(set) var awaitingConfirmation = false

    private let client: AccountClient?
    private let keychainAccount = "session"

    public init(config: AccountConfig? = AccountConfig.load()) {
        self.config = config
        client = config.map(AccountClient.init)
        if let data = SessionKeychain.get(keychainAccount),
           let saved = try? JSONDecoder.link.decode(AccountSession.self, from: data) {
            session = saved
        }
    }

    public var isConfigured: Bool { config != nil }
    public var isSignedIn: Bool { session != nil }

    public func signIn(email: String, password: String) async {
        await perform { try await $0.signIn(email: email, password: password) }
    }

    public func signUp(email: String, password: String) async {
        guard let client else { return }
        busy = true
        error = nil
        awaitingConfirmation = false
        do {
            if let session = try await client.signUp(email: email, password: password) {
                store(session)
            } else {
                awaitingConfirmation = true
            }
        } catch {
            self.error = error.localizedDescription
        }
        busy = false
    }

    public func signOut() {
        session = nil
        SessionKeychain.remove(keychainAccount)
    }

    /// A session that is valid for at least a minute, refreshed if needed.
    public func validSession() async -> AccountSession? {
        guard let session, let client else { return nil }
        guard session.isExpiring else { return session }
        if let fresh = try? await client.refresh(session) {
            store(fresh)
            return fresh
        }
        return nil
    }

    /// The user id a phone's access token belongs to (for the Mac station).
    public func userID(for accessToken: String) async -> String? {
        await client?.userID(for: accessToken)
    }

    private func perform(_ work: (AccountClient) async throws -> AccountSession) async {
        guard let client else { return }
        busy = true
        error = nil
        do {
            store(try await work(client))
        } catch {
            self.error = error.localizedDescription
        }
        busy = false
    }

    private func store(_ session: AccountSession) {
        self.session = session
        if let data = try? JSONEncoder.link.encode(session) { SessionKeychain.set(data, for: keychainAccount) }
    }
}

enum SessionKeychain {
    private static let service = "com.oasisspaces.account"

    static func set(_ value: Data, for account: String) {
        remove(account)
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecAttrAccount as String: account, kSecValueData as String: value,
                                    kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock]
        SecItemAdd(query as CFDictionary, nil)
    }

    static func get(_ account: String) -> Data? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecAttrAccount as String: account, kSecReturnData as String: true]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess else { return nil }
        return item as? Data
    }

    static func remove(_ account: String) {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecAttrAccount as String: account]
        SecItemDelete(query as CFDictionary)
    }
}
