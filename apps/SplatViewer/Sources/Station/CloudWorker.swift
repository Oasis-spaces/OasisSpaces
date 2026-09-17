import Foundation
import OasisLink

/// The Mac's side of the cloud relay: when signed in, it watches the
/// account's queued jobs, takes one at a time, fetches the capture (parts
/// reassembled), runs the pipeline as for a local job, and sends the results
/// back up. The phone that sent the job sees the same progress it would on
/// the local network.
@MainActor
final class CloudWorker {
    private let cloud: CloudLink
    private let runner: PipelineRunner
    private let folder: URL
    private(set) var jobs: [Job] = []
    private(set) var problem: String?
    /// Result files of finished cloud jobs, by job id, on this Mac.
    private var resultPaths: [String: [String: URL]] = [:]
    var onChange: (() -> Void)?
    private var loop: Task<Void, Never>?
    private var busy = false
    /// Results above this are not sent up (the storage's object limit).
    static let resultLimit: Int64 = 50 * 1024 * 1024
    static let pollSeconds: UInt64 = 20

    init(cloud: CloudLink, runner: PipelineRunner, folder: URL) {
        self.cloud = cloud
        self.runner = runner
        self.folder = folder
        if let data = try? Data(contentsOf: folder.appendingPathComponent("results.json")),
           let saved = try? JSONDecoder().decode([String: [String: URL]].self, from: data) {
            resultPaths = saved
        }
    }

    func start() {
        guard loop == nil else { return }
        loop = Task { [weak self] in
            while !Task.isCancelled {
                await self?.tick()
                try? await Task.sleep(nanoseconds: Self.pollSeconds * 1_000_000_000)
            }
        }
    }

    func stop() {
        loop?.cancel()
        loop = nil
    }

    func resultURL(job id: String, name: String) -> URL? {
        guard let url = resultPaths[id]?[name], FileManager.default.fileExists(atPath: url.path) else { return nil }
        return url
    }

    private func tick() async {
        guard cloud.isAvailable, let client = await cloud.client() else {
            if !jobs.isEmpty { jobs = []; onChange?() }
            return
        }
        do {
            jobs = try await client.jobs()
            problem = nil
        } catch {
            problem = "Cloud: \(error.localizedDescription)"
            onChange?()
            return
        }
        onChange?()
        guard !busy, let next = jobs.first(where: { $0.status == .queued }) else { return }
        busy = true
        await process(next, client)
        busy = false
        jobs = (try? await client.jobs()) ?? jobs
        onChange?()
    }

    private func process(_ job: Job, _ client: StationClient) async {
        // Another Mac on the account may be first: the claim decides.
        guard let claimed = try? await client.claim(job: job.id, station: StationService.computerName) else { return }
        let inputs = folder.appendingPathComponent(job.id, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: inputs, withIntermediateDirectories: true)
            for file in claimed.filesReceived {
                let target = inputs.appendingPathComponent(file)
                try? FileManager.default.removeItem(at: target)
                for url in try await client.downloadURLs(job: job.id, file: file) {
                    try await client.fetch(url, appendingTo: target)
                }
            }
            try? await client.deleteInputs(job: job.id)
            _ = try await client.report(job: job.id, .init(stageIndex: 0, message: "Received on \(StationService.computerName)"))
        } catch {
            _ = try? await client.report(job: job.id, .init(message: "Could not fetch the capture: \(error.localizedDescription)", status: .failed))
            return
        }
        jobs = (try? await client.jobs()) ?? jobs
        onChange?()

        let outcome = await runner.run(job: claimed, inputs: inputs) { index, note in
            Task { @MainActor in
                _ = try? await client.report(job: job.id, .init(stageIndex: index, message: note))
                StationService.shared.refresh()
            }
        }
        var sent: [ResultFile] = []
        var paths: [String: URL] = [:]
        for (file, url) in outcome.results {
            paths[file.name] = url
            guard file.bytes <= Self.resultLimit else { continue }
            do {
                let signed = try await client.resultUploadURL(job: job.id, name: file.name)
                try await client.put(try Data(contentsOf: url), to: signed) { _ in }
                sent.append(file)
            } catch {
                continue   // the phone gets what made it up; the Mac keeps everything
            }
        }
        resultPaths[job.id] = paths
        save()
        _ = try? await client.report(job: job.id, .init(message: outcome.message, status: outcome.ok ? .done : .failed, results: sent))
    }

    private func save() {
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(resultPaths) {
            try? data.write(to: folder.appendingPathComponent("results.json"))
        }
    }
}
