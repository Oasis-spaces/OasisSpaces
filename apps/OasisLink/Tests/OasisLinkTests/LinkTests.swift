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

/// A stand-in for Supabase Auth: two users, fixed passwords.
private func fakeAuthServer() throws -> HTTPServer {
    let users = ["mac@example.com": ("user-1", "right"), "other@example.com": ("user-2", "right")]
    return try HTTPServer(port: 0) { request in
        switch (request.method, request.path) {
        case ("POST", "/auth/v1/token"):
            return .body(maxBytes: 4096) { data in
                let body = (try? JSONSerialization.jsonObject(with: data) as? [String: String]) ?? [:]
                guard request.headers["apikey"] == "anon", let email = body["email"], let user = users[email],
                      body["password"] == user.1 else {
                    return .json(["error": "invalid_grant", "error_description": "Invalid login credentials"], status: 400)
                }
                let json = "{\"access_token\":\"token-\(user.0)\",\"refresh_token\":\"r\",\"expires_in\":3600,\"user\":{\"id\":\"\(user.0)\",\"email\":\"\(email)\"}}"
                return HTTPResponse(status: 200, headers: ["Content-Type": "application/json"], body: .data(Data(json.utf8)))
            }
        case ("GET", "/auth/v1/user"):
            guard let token = request.bearerToken, token.hasPrefix("token-") else { return .respond(.error(401, "bad jwt")) }
            let json = "{\"id\":\"\(token.dropFirst(6))\"}"
            return .respond(HTTPResponse(status: 200, headers: ["Content-Type": "application/json"], body: .data(Data(json.utf8))))
        default:
            return .respond(.error(404, "no"))
        }
    }
}

final class AccountTests: XCTestCase {
    func testPhoneOnSameAccountPairsWithoutCode() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("oasisaccount-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let auth = try fakeAuthServer()
        let authReady = expectation(description: "auth")
        auth.start { if case .ready = $0 { authReady.fulfill() } }
        await fulfillment(of: [authReady], timeout: 5)
        defer { auth.stop() }
        let accounts = AccountClient(config: AccountConfig(url: URL(string: "http://127.0.0.1:\(auth.port!)")!, anonKey: "anon"))

        do {
            _ = try await accounts.signIn(email: "mac@example.com", password: "wrong")
            XCTFail("wrong password accepted")
        } catch let error as AccountError {
            XCTAssertEqual(error.message, "Invalid login credentials")
        }
        let macSession = try await accounts.signIn(email: "mac@example.com", password: "right")
        XCTAssertEqual(macSession.userID, "user-1")

        let station = Station(folder: folder, info: StationInfo(id: "mac", name: "Mac"), stages: ["a"],
                              runner: NoRunner())
        let server = try HTTPServer(port: 0) { station.route($0) }
        let ready = expectation(description: "station")
        server.start { if case .ready = $0 { ready.fulfill() } }
        await fulfillment(of: [ready], timeout: 5)
        defer { server.stop() }
        let base = URL(string: "http://127.0.0.1:\(server.port!)")!

        // Not signed in yet: account pairing is refused.
        do {
            _ = try await StationClient(baseURL: base).pair(account: macSession, device: "iPhone")
            XCTFail("paired before the Mac signed in")
        } catch let error as LinkError {
            XCTAssertEqual(error.status, 409)
        }

        station.setAccount(userID: macSession.userID) { await accounts.userID(for: $0) }
        let info = try await StationClient(baseURL: base).info()
        XCTAssertEqual(info.owner, macSession.ownerTag)

        let other = try await accounts.signIn(email: "other@example.com", password: "right")
        do {
            _ = try await StationClient(baseURL: base).pair(account: other, device: "Someone else's iPhone")
            XCTFail("another account paired")
        } catch let error as LinkError {
            XCTAssertEqual(error.status, 403)
        }

        let phoneSession = try await accounts.signIn(email: "mac@example.com", password: "right")
        let phone = StationClient(baseURL: base)
        _ = try await phone.pair(account: phoneSession, device: "My iPhone")
        let jobs = try await phone.jobs()
        XCTAssertEqual(jobs.count, 0)
        XCTAssertEqual(station.pairedDevices, ["My iPhone"])
    }
}

private final class NoRunner: JobRunner, @unchecked Sendable {
    func run(job: Job, inputs: URL, stage: @escaping @Sendable (Int, String?) -> Void) async -> JobOutcome {
        JobOutcome(ok: true)
    }
}

final class CloudModelTests: XCTestCase {
    /// The relay's JSON (Postgres dates with offsets and microseconds, the
    /// cloud flag and parts) decodes into the same Job the Mac sends.
    func testRelayJobDecodes() throws {
        let json = """
        {"id": "ab12cd34", "name": "Bedroom", "createdAt": "2026-09-18T02:40:12.123456+00:00",
         "capturedAt": "2026-09-18T02:39:00Z", "status": "queued", "stages": ["reconstruct", "densify", "shapes", "splat"],
         "stageIndex": 0, "message": null, "filesExpected": ["capture.json", "video.mov"],
         "filesReceived": ["capture.json", "video.mov"], "results": [], "parts": {"video.mov": 4, "capture.json": 1},
         "station": null, "cloud": true}
        """
        let job = try JSONDecoder.link.decode(Job.self, from: Data(json.utf8))
        XCTAssertTrue(job.cloud)
        XCTAssertEqual(job.parts["video.mov"], 4)
        XCTAssertEqual(job.status, .queued)
        var utc = Calendar(identifier: .gregorian)
        utc.timeZone = TimeZone(identifier: "UTC")!
        XCTAssertEqual(utc.component(.minute, from: job.createdAt), 40)
        XCTAssertEqual(utc.component(.hour, from: job.createdAt), 2)
        // A Mac station's job (no cloud fields) still decodes, as before.
        let local = try JSONDecoder.link.decode(Job.self, from: try JSONEncoder.link.encode(
            Job(id: "x", name: "Hall", createdAt: Date(), capturedAt: Date(), status: .done, stages: [], filesExpected: [])))
        XCTAssertFalse(local.cloud)
        XCTAssertTrue(local.parts.isEmpty)
    }

    func testCloudConfigIsBundled() {
        let config = CloudConfig.load()
        XCTAssertEqual(config?.url.host, "oasis-relay.onrender.com")
    }
}
