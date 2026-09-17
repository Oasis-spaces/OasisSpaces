import AppKit
import Foundation
import Observation
import OasisLink
import SystemConfiguration

/// The Mac end of the phone link: advertises itself on the local network,
/// accepts recordings from paired phones, analyses them one at a time and
/// serves the results back.
@MainActor
@Observable
final class StationService {
    static let shared = StationService()

    private(set) var jobs: [Job] = []
    private(set) var pairingCode = ""
    private(set) var pairedDevices: [String] = []
    private(set) var listening = false
    private(set) var problem: String?

    var repositoryPath: String {
        didSet {
            UserDefaults.standard.set(repositoryPath, forKey: "station.repository")
            runner.repository = URL(fileURLWithPath: repositoryPath)
        }
    }

    @ObservationIgnored private var server: HTTPServer?
    @ObservationIgnored private var station: Station?
    @ObservationIgnored private let runner: PipelineRunner

    private init() {
        let repository = UserDefaults.standard.string(forKey: "station.repository") ?? NSHomeDirectory() + "/OasisSpaces"
        let python = ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"]
            .first { FileManager.default.isExecutableFile(atPath: $0) } ?? "/usr/bin/python3"
        runner = PipelineRunner(repository: URL(fileURLWithPath: repository), python: URL(fileURLWithPath: python))
        repositoryPath = repository
    }

    func start() {
        guard server == nil else { return }
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Splat Viewer/Station", isDirectory: true)
        let station = Station(folder: support, info: StationInfo(id: Self.stationID, name: Self.computerName),
                              stages: PipelineRunner.stages, runner: runner)
        station.onChange = { jobs in
            Task { @MainActor in StationService.shared.jobs = jobs }
        }
        self.station = station
        do {
            let host = Self.localHostName
            let server = try HTTPServer(
                port: Link.defaultPort,
                service: (name: Self.computerName, type: Link.serviceType,
                          txt: ["id": Self.stationID, "name": Self.computerName, "host": host,
                                "port": String(Link.defaultPort)])) { station.route($0) }
            server.start { state in
                Task { @MainActor in
                    switch state {
                    case .ready:
                        StationService.shared.listening = true
                        StationService.shared.problem = nil
                    case .failed(let error):
                        StationService.shared.listening = false
                        StationService.shared.problem = "Could not listen on port \(Link.defaultPort): \(error.localizedDescription)"
                    default: break
                    }
                }
            }
            self.server = server
        } catch {
            problem = "Could not start the phone link: \(error.localizedDescription)"
        }
        refresh()
        station.resume()
    }

    func refresh() {
        guard let station else { return }
        jobs = station.allJobs
        pairingCode = station.pairingCode
        pairedDevices = station.pairedDevices
    }

    func newPairingCode() {
        station?.newPairingCode()
        refresh()
    }

    func unpairAll() {
        station?.unpairAll()
        refresh()
    }

    func runAgain(_ job: Job) {
        station?.retry(job.id)
        refresh()
    }

    func resultURL(_ job: Job, _ name: String) -> URL? {
        station?.resultURL(job: job.id, name: name)
    }

    /// Opens the job's splat (the filled one when there is one) as a tab.
    func open(_ job: Job) {
        let splat = job.results.first { $0.name == "splat-filled.splat" } ?? job.results.first { $0.kind == .splat }
        guard let splat, let url = resultURL(job, splat.name) else { return }
        Library.shared.add([url])
    }

    func reveal(_ job: Job) {
        guard let any = job.results.first, let url = resultURL(job, any.name) else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    // MARK: Identity

    private static var stationID: String {
        if let id = UserDefaults.standard.string(forKey: "station.id") { return id }
        let id = UUID().uuidString
        UserDefaults.standard.set(id, forKey: "station.id")
        return id
    }

    static var computerName: String {
        Host.current().localizedName ?? "Mac"
    }

    /// The Bonjour host name phones reach this Mac at, e.g. Sanskars-MacBook-Air.local.
    static var localHostName: String {
        if let name = SCDynamicStoreCopyLocalHostName(nil) as String? { return name + ".local" }
        return ProcessInfo.processInfo.hostName
    }
}
