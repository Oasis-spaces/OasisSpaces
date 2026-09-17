import Foundation
import Network

/// A Mac station seen on the local network.
public struct DiscoveredStation: Identifiable, Hashable, Sendable {
    public var id: String
    public var name: String
    /// http://<host>.local:<port>, from the Bonjour TXT record.
    public var baseURL: URL
    /// AccountSession.tag of the account the Mac is signed in to, if any.
    public var owner: String?
}

/// Watches the local network for Mac stations advertising Link.serviceType.
public final class StationBrowser: @unchecked Sendable {
    private let browser: NWBrowser
    private let queue = DispatchQueue(label: "oasislink.browser")

    public init() {
        let parameters = NWParameters()
        parameters.includePeerToPeer = true
        browser = NWBrowser(for: .bonjourWithTXTRecord(type: Link.serviceType, domain: nil), using: parameters)
    }

    /// `update` gets the full list on every change (on the main queue).
    public func start(update: @escaping @MainActor @Sendable ([DiscoveredStation]) -> Void) {
        browser.browseResultsChangedHandler = { results, _ in
            let stations = results.compactMap { result -> DiscoveredStation? in
                guard case .bonjour(let txt) = result.metadata,
                      let id = txt["id"], let host = txt["host"], let port = txt["port"],
                      let url = URL(string: "http://\(host):\(port)") else { return nil }
                return DiscoveredStation(id: id, name: txt["name"] ?? host, baseURL: url, owner: txt["owner"])
            }
            Task { @MainActor in update(stations) }
        }
        browser.start(queue: queue)
    }

    public func stop() {
        browser.cancel()
    }
}
