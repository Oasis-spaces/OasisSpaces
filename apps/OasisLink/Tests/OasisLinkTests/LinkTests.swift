import XCTest
import CryptoKit
@testable import OasisLink

/// Pretends to be the pipeline: three stages, then one splat and one image.
private final class FakeRunner: JobRunner, @unchecked Sendable {
    let output: URL
    var seenInputs: [String] = []
    init(output: URL) { self.output = output }

    func run(job: Job, inputs: URL, stage: @escaping @Sendable (Int, String?) -> Void) async -> JobOutcome {
        seenInputs = (try? FileManager.default.contentsOfDirectory(atPath: inputs.path))?.sorted() ?? []
        for i in 0..<3 {
            stage(i, "stage \(i)")
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        let splat = output.appendingPathComponent("\(job.id).splat")
        let image = output.appendingPathComponent("\(job.id).png")
        try? Data(repeating: 7, count: 3_000_000).write(to: splat)
        try? Data("png".utf8).write(to: image)
        return JobOutcome(ok: true, message: "done", results: [
            (ResultFile(name: "room.splat", kind: .splat, bytes: 3_000_000), splat),
            (ResultFile(name: "room-render.png", kind: .image, bytes: 3), image),
        ])
    }
}

final class LinkTests: XCTestCase {
    private var folder: URL!
    private var server: HTTPServer!
    private var station: Station!
    private var runner: FakeRunner!

    override func setUp() async throws {
        folder = FileManager.default.temporaryDirectory.appendingPathComponent("oasislink-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        runner = FakeRunner(output: folder)
        station = Station(folder: folder.appendingPathComponent("station"),
                          info: StationInfo(id: "test-mac", name: "Test Mac"),
                          stages: ["reconstruct", "densify", "shapes"], runner: runner)
        let station = self.station!
        server = try HTTPServer(port: 0) { station.route($0) }
        let ready = expectation(description: "listening")
        server.start { state in if case .ready = state { ready.fulfill() } }
        await fulfillment(of: [ready], timeout: 5)
    }

    override func tearDown() {
        server.stop()
        try? FileManager.default.removeItem(at: folder)
    }

    private var client: StationClient {
        StationClient(baseURL: URL(string: "http://127.0.0.1:\(server.port!)")!)
    }

    func testUnpairedPhoneIsRefusedAndWrongCodeFails() async throws {
        let c = client
        let info = try await c.info()
        XCTAssertEqual(info.name, "Test Mac")
        do {
            _ = try await c.jobs()
            XCTFail("jobs without pairing")
        } catch let error as LinkError {
            XCTAssertEqual(error.status, 401)
        }
        do {
            _ = try await c.pair(code: "000000" == station.pairingCode ? "111111" : "000000", device: "iPhone")
            XCTFail("wrong code accepted")
        } catch let error as LinkError {
            XCTAssertEqual(error.status, 403)
        }
    }

    func testFullRoundTrip() async throws {
        let c = client
        let pairing = try await c.pair(code: station.pairingCode, device: "Test iPhone")
        XCTAssertEqual(pairing.station.id, "test-mac")
        XCTAssertEqual(station.pairedDevices, ["Test iPhone"])

        // A recording: a 25 MB "video" and two small files.
        let capture = folder.appendingPathComponent("capture")
        try FileManager.default.createDirectory(at: capture, withIntermediateDirectories: true)
        var bytes = Data(count: 25_000_000)
        bytes.withUnsafeMutableBytes { raw in
            for i in stride(from: 0, to: raw.count, by: 4096) { raw[i] = UInt8(truncatingIfNeeded: i &* 31) }
        }
        try bytes.write(to: capture.appendingPathComponent("video.mov"))
        try Data("{}".utf8).write(to: capture.appendingPathComponent("capture.json"))
        try Data("{\"t\":0}\n".utf8).write(to: capture.appendingPathComponent("frames.jsonl"))

        let job = try await c.createJob(NewJob(name: "Bedroom", files: Link.captureFiles, capturedAt: Date()))
        XCTAssertEqual(job.status, .receiving)
        do {
            _ = try await c.startJob(job.id)
            XCTFail("started with files missing")
        } catch let error as LinkError {
            XCTAssertEqual(error.status, 409)
        }

        let reported = ProgressBox()
        for name in Link.captureFiles {
            try await c.upload(file: capture.appendingPathComponent(name), to: job.id, as: name) { reported.set($0) }
        }
        XCTAssertEqual(reported.value, 1.0, accuracy: 0.001)
        let stored = station.inputFolder(job.id).appendingPathComponent("video.mov")
        XCTAssertEqual(try Data(contentsOf: stored), bytes, "the video must arrive byte for byte")

        _ = try await c.startJob(job.id)
        var finished = try await c.job(job.id)
        for _ in 0..<100 where finished.status != .done && finished.status != .failed {
            try await Task.sleep(nanoseconds: 50_000_000)
            finished = try await c.job(job.id)
        }
        XCTAssertEqual(finished.status, .done)
        XCTAssertEqual(finished.progress, 1)
        XCTAssertEqual(runner.seenInputs, ["capture.json", "frames.jsonl", "video.mov"])

        let splatResult = finished.results.first { $0.kind == .splat }!
        let downloaded = try await c.download(splatResult, of: job.id, into: folder.appendingPathComponent("phone"))
        XCTAssertEqual(try Data(contentsOf: downloaded), Data(repeating: 7, count: 3_000_000))

        let all = try await c.jobs()
        XCTAssertEqual(all.map(\.id), [job.id])
    }

    func testUploadRejectsUnexpectedAndUnsafeNames() async throws {
        let c = client
        _ = try await c.pair(code: station.pairingCode, device: "iPhone")
        let job = try await c.createJob(NewJob(name: "x", files: ["video.mov", "../evil"], capturedAt: Date()))
        XCTAssertEqual(job.filesExpected, ["video.mov"])
        let file = folder.appendingPathComponent("f")
        try Data("x".utf8).write(to: file)
        do {
            try await c.upload(file: file, to: job.id, as: "other.bin") { _ in }
            XCTFail("unexpected file accepted")
        } catch let error as LinkError {
            XCTAssertEqual(error.status, 400)
        }
    }

    func testStationRemembersJobsAndPairingAcrossRestarts() async throws {
        let c = client
        _ = try await c.pair(code: station.pairingCode, device: "iPhone")
        let job = try await c.createJob(NewJob(name: "Kitchen", files: ["video.mov"], capturedAt: Date()))
        let reopened = Station(folder: folder.appendingPathComponent("station"),
                               info: StationInfo(id: "test-mac", name: "Test Mac"),
                               stages: ["a"], runner: runner)
        XCTAssertEqual(reopened.allJobs.map(\.id), [job.id])
        XCTAssertEqual(reopened.pairedDevices, ["iPhone"])
    }
}

private final class ProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var v = 0.0
    func set(_ x: Double) { lock.withLock { v = max(v, x) } }
    var value: Double { lock.withLock { v } }
}
