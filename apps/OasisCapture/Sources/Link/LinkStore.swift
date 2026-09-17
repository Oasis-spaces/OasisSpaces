import Foundation
import Security
import OasisLink

/// The phone's side of the link to the Mac: finding Macs on the network,
/// pairing, sending recordings, following their analysis and keeping results.
@MainActor
final class LinkStore: ObservableObject {
    static let shared = LinkStore()

    struct PairedMac: Codable, Equatable {
        var id: String
        var name: String
        var baseURL: URL
    }

    struct Upload: Equatable {
        var recording: URL
        var progress: Double
        var step: String
        var error: String?
        var jobID: String?
    }

    @Published private(set) var discovered: [DiscoveredStation] = []
    @Published private(set) var paired: PairedMac?
    @Published private(set) var jobs: [Job] = []
    @Published private(set) var reachable = false
    @Published private(set) var lastError: String?
    @Published var upload: Upload?
    let account = AccountStore()

    private let browser = StationBrowser()
    private var browsing = false
    private var polling: Task<Void, Never>?

    private init() {
        if let data = UserDefaults.standard.data(forKey: "link.paired"),
           let mac = try? JSONDecoder().decode(PairedMac.self, from: data) {
            paired = mac
        }
    }

    // MARK: Finding and pairing

    func startBrowsing() {
        guard !browsing else { return }
        browsing = true
        browser.start { [weak self] stations in
            guard let self else { return }
            self.discovered = stations
            // A paired Mac whose address changed (a new network name): follow it.
            if var mac = self.paired, let seen = stations.first(where: { $0.id == mac.id }), seen.baseURL != mac.baseURL {
                mac.baseURL = seen.baseURL
                self.save(mac)
            }
        }
    }

    func pair(with station: DiscoveredStation, code: String) async -> Bool {
        let client = StationClient(baseURL: station.baseURL)
        do {
            let response = try await client.pair(code: code.filter(\.isNumber), device: UIDeviceName.current)
            Keychain.set(response.token, for: response.station.id)
            save(PairedMac(id: response.station.id, name: response.station.name, baseURL: station.baseURL))
            lastError = nil
            await refresh()
            return true
        } catch {
            lastError = error.localizedDescription
            return false
        }
    }

    /// Pairs with a Mac signed in to the same account, no code needed.
    func pairByAccount(with station: DiscoveredStation) async -> Bool {
        guard let session = await account.validSession() else {
            lastError = "Sign in first"
            return false
        }
        let client = StationClient(baseURL: station.baseURL)
        do {
            let response = try await client.pair(account: session, device: UIDeviceName.current)
            Keychain.set(response.token, for: response.station.id)
            save(PairedMac(id: response.station.id, name: response.station.name, baseURL: station.baseURL))
            lastError = nil
            await refresh()
            return true
        } catch {
            lastError = error.localizedDescription
            return false
        }
    }

    /// Macs on this network signed in to the same account as this phone.
    var macsOnMyAccount: [DiscoveredStation] {
        guard let tag = account.session?.ownerTag else { return [] }
        return discovered.filter { $0.owner == tag }
    }

    func unpair() {
        if let mac = paired { Keychain.remove(mac.id) }
        paired = nil
        jobs = []
        UserDefaults.standard.removeObject(forKey: "link.paired")
    }

    private func save(_ mac: PairedMac) {
        paired = mac
        UserDefaults.standard.set(try? JSONEncoder().encode(mac), forKey: "link.paired")
    }

    private var client: StationClient? {
        guard let mac = paired, let token = Keychain.get(mac.id) else { return nil }
        return StationClient(baseURL: mac.baseURL, token: token)
    }

    // MARK: Jobs

    func refresh() async {
        guard let client else { return }
        do {
            jobs = try await client.jobs()
            reachable = true
            lastError = nil
        } catch let error as LinkError where error.status == 401 {
            reachable = true
            lastError = "The Mac no longer knows this phone. Pair again."
        } catch {
            reachable = false
        }
    }

    /// Refreshes every few seconds while something is watching.
    func startPolling() {
        guard polling == nil else { return }
        polling = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refresh()
                try? await Task.sleep(nanoseconds: 3_000_000_000)
            }
        }
    }

    func stopPolling() {
        polling?.cancel()
        polling = nil
    }

    /// Sends a recording folder (video.mov, frames.jsonl, capture.json) to the
    /// Mac and starts its analysis.
    func send(recording folder: URL, name: String) async {
        guard let client else {
            upload = Upload(recording: folder, progress: 0, step: "", error: "Pair with your Mac first")
            return
        }
        let files = Link.captureFiles.filter { FileManager.default.fileExists(atPath: folder.appendingPathComponent($0).path) }
        guard files.contains("video.mov") else {
            upload = Upload(recording: folder, progress: 0, step: "", error: "This recording has no video")
            return
        }
        let sizes = files.map { (try? FileManager.default.attributesOfItem(atPath: folder.appendingPathComponent($0).path)[.size] as? Int64) ?? 0 }
        let total = max(1, sizes.reduce(0, +))
        upload = Upload(recording: folder, progress: 0, step: "Connecting to the Mac")
        do {
            let capturedAt = (try? folder.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date()
            let job = try await client.createJob(NewJob(name: name, files: files, capturedAt: capturedAt))
            upload?.jobID = job.id
            var done: Int64 = 0
            for (file, size) in zip(files, sizes) {
                upload?.step = file == "video.mov" ? "Sending the video" : "Sending \(file)"
                let before = done
                try await client.upload(file: folder.appendingPathComponent(file), to: job.id, as: file) { fraction in
                    let overall = Double(before + Int64(Double(size) * fraction)) / Double(total)
                    Task { @MainActor in LinkStore.shared.upload?.progress = overall }
                }
                done += size
            }
            upload?.step = "Starting the analysis"
            _ = try await client.startJob(job.id)
            upload?.progress = 1
            upload?.step = "Sent. The Mac is analysing it."
            markSent(folder, jobID: job.id)
            await refresh()
        } catch {
            upload?.error = error.localizedDescription
        }
    }

    // MARK: Results

    /// Where a job's results are kept on the phone.
    func resultsFolder(_ job: Job) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Results/\(job.id)", isDirectory: true)
    }

    func localResult(_ job: Job, _ name: String) -> URL? {
        let url = resultsFolder(job).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Downloads the job's results that are not on the phone yet.
    func fetchResults(_ job: Job) async {
        guard let client else { return }
        for result in job.results where localResult(job, result.name) == nil {
            _ = try? await client.download(result, of: job.id, into: resultsFolder(job))
        }
        objectWillChange.send()
    }

    // MARK: Recordings on the phone

    func recordings() -> [URL] {
        let captures = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Captures", isDirectory: true)
        let folders = (try? FileManager.default.contentsOfDirectory(at: captures, includingPropertiesForKeys: [.creationDateKey])) ?? []
        return folders.filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("video.mov").path) }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
    }

    func sentJobID(_ folder: URL) -> String? {
        try? String(contentsOf: folder.appendingPathComponent(".sent"), encoding: .utf8)
    }

    private func markSent(_ folder: URL, jobID: String) {
        try? jobID.write(to: folder.appendingPathComponent(".sent"), atomically: true, encoding: .utf8)
    }
}

enum UIDeviceName {
    @MainActor static var current: String {
        UIDevice.current.name
    }
}

/// The pairing token, kept in the Keychain.
enum Keychain {
    private static let service = "com.oasisspaces.capture.link"

    static func set(_ value: String, for account: String) {
        remove(account)
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecAttrAccount as String: account, kSecValueData as String: Data(value.utf8),
                                    kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock]
        SecItemAdd(query as CFDictionary, nil)
    }

    static func get(_ account: String) -> String? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecAttrAccount as String: account, kSecReturnData as String: true]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess, let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func remove(_ account: String) {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                    kSecAttrAccount as String: account]
        SecItemDelete(query as CFDictionary)
    }
}

import UIKit
