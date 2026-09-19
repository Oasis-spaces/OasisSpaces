import Foundation
import OasisLink

/// Runs the OasisSpaces pipeline on a recording from the phone: each stage of
/// pipeline/agent.py in turn, stopping when a stage's gate says stop, then
/// reports the viewer files, renders and report as results.
final class PipelineRunner: JobRunner, @unchecked Sendable {
    static let stages = ["reconstruct", "densify", "shapes", "splat"]

    /// The OasisSpaces checkout and the Python that runs it.
    var repository: URL
    var python: URL

    init(repository: URL, python: URL) {
        self.repository = repository
        self.python = python
    }

    func run(job: Job, inputs: URL, stage: @escaping @Sendable (Int, String?) -> Void) async -> JobOutcome {
        let agent = repository.appendingPathComponent("pipeline/agent.py")
        guard FileManager.default.fileExists(atPath: agent.path) else {
            return JobOutcome(ok: false, message: "No OasisSpaces checkout at \(repository.path). Set it in Phone captures.")
        }
        let space = spaceName(job)
        let spaceFolder = repository.appendingPathComponent("spaces/\(space)")
        // The best-splat judge groups splats by the video's file name, and every
        // phone recording is video.mov: give each its own name (a hard link, no copy).
        let video = inputs.appendingPathComponent("\(space).mov")
        if !FileManager.default.fileExists(atPath: video.path) {
            do {
                try FileManager.default.linkItem(at: inputs.appendingPathComponent("video.mov"), to: video)
            } catch {
                try? FileManager.default.copyItem(at: inputs.appendingPathComponent("video.mov"), to: video)
            }
        }
        let log = repository.appendingPathComponent("runs/\(space)-station.log")
        try? FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)

        for (index, name) in Self.stages.enumerated() {
            stage(index, Self.describe(name))
            let status = await runStage(name, video: video, space: space, log: log)
            let gate = gate(of: name, in: spaceFolder)
            if gate?.status == "stop" || (gate == nil && status != 0) {
                return JobOutcome(ok: false, message: "\(Self.describe(name)) stopped: \(gate?.why ?? "see \(log.path)")",
                                  results: results(in: spaceFolder))
            }
            // The phone's own files beside the pipeline's, once the space exists.
            if index == 0 { keepCaptureFiles(from: inputs, in: spaceFolder) }
        }
        let scene = spaceFolder.appendingPathComponent("scene")
        let hasScene = FileManager.default.fileExists(atPath: scene.appendingPathComponent("scene.json").path)
        return JobOutcome(ok: true, message: gate(of: "splat", in: spaceFolder)?.why, results: results(in: spaceFolder),
                          scene: hasScene ? scene : nil)
    }

    private func spaceName(_ job: Job) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmm"
        return "phone-\(formatter.string(from: job.capturedAt))-\(job.id)"
    }

    static func describe(_ stage: String) -> String {
        switch stage {
        case "reconstruct": return "Finding where the phone was"
        case "densify": return "Measuring depth and recognising objects"
        case "shapes": return "Building the room model"
        case "splat": return "Training the 3D splat"
        default: return stage
        }
    }

    private func runStage(_ stage: String, video: URL, space: String, log: URL) async -> Int32 {
        await withCheckedContinuation { continuation in
            let process = Process()
            process.executableURL = python
            process.arguments = [repository.appendingPathComponent("pipeline/agent.py").path, video.path,
                                 "--name", space, "--stage", stage]
            process.currentDirectoryURL = repository
            var environment = ProcessInfo.processInfo.environment
            // COLMAP, ffmpeg, Blender and claude live here; apps launched from Finder get a bare PATH.
            environment["PATH"] = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin",
                                   NSHomeDirectory() + "/.local/bin", environment["PATH"] ?? ""].joined(separator: ":")
            process.environment = environment
            FileManager.default.createFile(atPath: log.path, contents: nil)
            if let handle = try? FileHandle(forWritingTo: log) {
                handle.seekToEndOfFile()
                handle.write(Data("\n===== \(Date()) stage \(stage)\n".utf8))
                process.standardOutput = handle
                process.standardError = handle
            }
            process.terminationHandler = { continuation.resume(returning: $0.terminationStatus) }
            do {
                try process.run()
            } catch {
                continuation.resume(returning: -1)
            }
        }
    }

    private func gate(of stage: String, in space: URL) -> (status: String, why: String)? {
        guard let data = try? Data(contentsOf: space.appendingPathComponent("agent-report.json")),
              let report = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let gates = report["gates"] as? [String: Any],
              let gate = gates[stage] as? [String: Any],
              let status = gate["status"] as? String else { return nil }
        return (status, gate["why"] as? String ?? "")
    }

    private func keepCaptureFiles(from inputs: URL, in space: URL) {
        let target = space.appendingPathComponent("phone-capture")
        try? FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        for name in ["capture.json", "frames.jsonl"] {
            let destination = target.appendingPathComponent(name)
            if !FileManager.default.fileExists(atPath: destination.path) {
                try? FileManager.default.copyItem(at: inputs.appendingPathComponent(name), to: destination)
            }
        }
    }

    /// What the phone can fetch: the viewer files (the filled splat when its
    /// fill was kept), the room renders and the report.
    private func results(in space: URL) -> [(file: ResultFile, url: URL)] {
        let candidates: [(String, ResultFile.Kind)] = [
            ("splat.splat", .splat), ("splat.view.json", .view),
            ("splat-filled.splat", .splat), ("splat-filled.view.json", .view),
            ("room-render.png", .image), ("room-render-plan.png", .image),
            ("agent-report.json", .report),
        ]
        return candidates.compactMap { name, kind in
            let url = space.appendingPathComponent(name)
            guard let size = try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64 else { return nil }
            return (ResultFile(name: name, kind: kind, bytes: size), url)
        }
    }
}
