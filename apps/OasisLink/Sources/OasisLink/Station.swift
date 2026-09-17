import Foundation

/// Runs the analysis of one recording; supplied by the Mac app.
public protocol JobRunner: AnyObject, Sendable {
    /// Runs the pipeline on the recording in `inputs`. Call `stage` as each
    /// stage starts (its index, and a note). Returns the result files.
    func run(job: Job, inputs: URL, stage: @escaping @Sendable (Int, String?) -> Void) async -> JobOutcome
}

public struct JobOutcome: Sendable {
    public var ok: Bool
    public var message: String?
    /// Result files and where they are on disk.
    public var results: [(file: ResultFile, url: URL)]

    public init(ok: Bool, message: String? = nil, results: [(file: ResultFile, url: URL)] = []) {
        self.ok = ok
        self.message = message
        self.results = results
    }
}

/// The Mac side of the link: pairing, accepting recordings, a queue that runs
/// them one at a time, and serving results. State lives under `folder`:
/// tokens.json, and jobs/<id>/{job.json, results.json, input/}.
public final class Station: @unchecked Sendable {
    public var info: StationInfo { lock.withLock { var i = baseInfo; i.owner = owner.map(AccountSession.tag); return i } }
    public let stages: [String]
    /// Called on every change, with every job, newest first (any queue).
    public var onChange: (@Sendable ([Job]) -> Void)?

    private let baseInfo: StationInfo
    /// The account the Mac is signed in to, and how to tell whose a phone's login is.
    private var owner: String?
    private var accountCheck: (@Sendable (String) async -> String?)?
    private let folder: URL
    private let runner: JobRunner
    private let lock = NSLock()
    private var jobs: [String: Job] = [:]
    private var resultPaths: [String: [String: String]] = [:]   // job -> result name -> path
    private var tokens: [String: String] = [:]                    // token -> device
    private var code: String
    private var running = false

    public init(folder: URL, info: StationInfo, stages: [String], runner: JobRunner) {
        self.folder = folder
        self.baseInfo = info
        self.stages = stages
        self.runner = runner
        code = Station.newCode()
        try? FileManager.default.createDirectory(at: jobsFolder, withIntermediateDirectories: true)
        load()
    }

    // MARK: Pairing

    /// The 6-digit code a phone enters to pair.
    public var pairingCode: String { lock.withLock { code } }

    public func newPairingCode() {
        lock.withLock { code = Station.newCode() }
    }

    public var pairedDevices: [String] { lock.withLock { Array(tokens.values).sorted() } }

    /// Signs the station in to an account (nil signs out): phones signed in to the
    /// same account pair without a code. `check` returns the user id a phone's
    /// access token belongs to (AccountClient.userID).
    public func setAccount(userID: String?, check: (@Sendable (String) async -> String?)?) {
        lock.withLock {
            owner = userID
            accountCheck = check
        }
    }

    public func unpairAll() {
        lock.withLock { tokens = [:] }
        saveTokens()
    }

    private static func newCode() -> String {
        String(format: "%06d", Int.random(in: 0...999_999))
    }

    // MARK: Jobs

    public var allJobs: [Job] {
        lock.withLock { jobs.values.sorted { $0.createdAt > $1.createdAt } }
    }

    public func resultURL(job id: String, name: String) -> URL? {
        lock.withLock { resultPaths[id]?[name].map { URL(fileURLWithPath: $0) } }
    }

    public func inputFolder(_ id: String) -> URL {
        jobsFolder.appendingPathComponent(id).appendingPathComponent("input")
    }

    /// Runs a job again (after a failure, or with a newer pipeline).
    public func retry(_ id: String) {
        update(id) {
            guard $0.status == .failed || $0.status == .done else { return }
            $0.status = .queued
            $0.stageIndex = 0
            $0.message = nil
            $0.results = []
        }
        runNext()
    }

    // MARK: Routing

    public func route(_ request: HTTPRequest) -> RouteDecision {
        let s = request.segments
        switch (request.method, s.count) {
        case ("GET", 1) where s[0] == "info":
            return .respond(.json(info))

        case ("POST", 2) where s[0] == "pair" && s[1] == "account":
            return .asyncBody(maxBytes: 16384) { [weak self] data in
                guard let self, let pair = try? JSONDecoder.link.decode(AccountPairRequest.self, from: data) else {
                    return .error(400, "bad pairing request")
                }
                let (owner, check) = self.lock.withLock { (self.owner, self.accountCheck) }
                guard let owner, let check else { return .error(409, "The Mac is not signed in to an account") }
                guard let user = await check(pair.accessToken) else { return .error(401, "Sign in again on the phone") }
                guard user == owner else { return .error(403, "This phone is signed in to a different account than the Mac") }
                return .json(self.issueToken(device: pair.device))
            }

        case ("POST", 1) where s[0] == "pair":
            return .body(maxBytes: 4096) { [weak self] data in
                guard let self, let pair = try? JSONDecoder.link.decode(PairRequest.self, from: data) else {
                    return .error(400, "bad pairing request")
                }
                return self.pair(pair)
            }
        default:
            break
        }

        guard let token = request.bearerToken, lock.withLock({ tokens[token] != nil }) else {
            return .respond(.error(401, "Pair this phone with the Mac first"))
        }

        switch (request.method, s.count) {
        case ("GET", 1) where s[0] == "jobs":
            return .respond(.json(allJobs))

        case ("POST", 1) where s[0] == "jobs":
            return .body(maxBytes: 65536) { [weak self] data in
                guard let self, let new = try? JSONDecoder.link.decode(NewJob.self, from: data) else {
                    return .error(400, "bad job")
                }
                return .json(self.create(new), status: 201)
            }

        case ("GET", 2) where s[0] == "jobs":
            guard let job = lock.withLock({ jobs[s[1]] }) else { return .respond(.error(404, "no such job")) }
            return .respond(.json(job))

        case ("PUT", 4) where s[0] == "jobs" && s[2] == "files":
            let id = s[1], name = s[3]
            guard let job = lock.withLock({ jobs[id] }) else { return .respond(.error(404, "no such job")) }
            guard job.filesExpected.contains(name), Station.safeName(name) else {
                return .respond(.error(400, "unexpected file \(name)"))
            }
            guard job.status == .receiving else { return .respond(.error(409, "the job is past receiving files")) }
            let folder = inputFolder(id)
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let partial = folder.appendingPathComponent(name + ".part")
            let expected = request.contentLength
            return .upload(to: partial) { [weak self] complete in
                guard let self else { return .error(500, "station gone") }
                let size = (try? FileManager.default.attributesOfItem(atPath: partial.path)[.size] as? Int64) ?? -1
                guard complete, size == expected else {
                    try? FileManager.default.removeItem(at: partial)
                    return .error(400, "\(name) arrived incomplete")
                }
                let target = folder.appendingPathComponent(name)
                try? FileManager.default.removeItem(at: target)
                try? FileManager.default.moveItem(at: partial, to: target)
                self.update(id) { job in
                    if !job.filesReceived.contains(name) { job.filesReceived.append(name) }
                }
                return HTTPResponse(status: 204)
            }

        case ("POST", 3) where s[0] == "jobs" && s[2] == "start":
            let id = s[1]
            return .body(maxBytes: 4096) { [weak self] _ in
                guard let self, let job = self.lock.withLock({ self.jobs[id] }) else { return .error(404, "no such job") }
                let missing = job.filesExpected.filter { !job.filesReceived.contains($0) }
                guard missing.isEmpty else { return .error(409, "still missing \(missing.joined(separator: ", "))") }
                if job.status == .receiving {
                    self.update(id) { $0.status = .queued }
                    self.runNext()
                }
                return .json(self.lock.withLock { self.jobs[id]! })
            }

        case ("GET", 4) where s[0] == "jobs" && s[2] == "results":
            guard let url = resultURL(job: s[1], name: s[3]) else { return .respond(.error(404, "no such result")) }
            return .respond(HTTPResponse(status: 200, body: .file(url)))

        default:
            return .respond(.error(404, "unknown request"))
        }
    }

    private func issueToken(device: String) -> PairResponse {
        let token = UUID().uuidString + UUID().uuidString
        lock.withLock { tokens[token] = device }
        saveTokens()
        return PairResponse(token: token, station: info)
    }

    private func pair(_ request: PairRequest) -> HTTPResponse {
        let accepted: String? = lock.withLock {
            guard request.code == code else { return nil }
            let token = UUID().uuidString + UUID().uuidString
            tokens[token] = request.device
            code = Station.newCode()   // one pairing per code
            return token
        }
        guard let token = accepted else { return .error(403, "That code is not the one on the Mac") }
        saveTokens()
        return .json(PairResponse(token: token, station: info))
    }

    private func create(_ new: NewJob) -> Job {
        let id = String(UUID().uuidString.prefix(8)).lowercased()
        let job = Job(id: id, name: new.name, createdAt: Date(), capturedAt: new.capturedAt, status: .receiving,
                      stages: stages, filesExpected: new.files.filter(Station.safeName))
        lock.withLock { jobs[id] = job }
        save(job)
        notify()
        return job
    }

    /// Starts the next queued job if none is running.
    private func runNext() {
        let next: Job? = lock.withLock {
            guard !running else { return nil }
            let job = jobs.values.filter { $0.status == .queued }.min { $0.createdAt < $1.createdAt }
            if job != nil { running = true }
            return job
        }
        guard let job = next else { return }
        update(job.id) {
            $0.status = .running
            $0.stageIndex = 0
        }
        Task.detached { [weak self] in
            guard let self else { return }
            let outcome = await self.runner.run(job: job, inputs: self.inputFolder(job.id)) { index, note in
                self.update(job.id) {
                    $0.stageIndex = index
                    $0.message = note
                }
            }
            self.lock.withLock {
                self.resultPaths[job.id] = Dictionary(uniqueKeysWithValues: outcome.results.map { ($0.file.name, $0.url.path) })
            }
            self.saveResultPaths(job.id)
            self.update(job.id) {
                $0.status = outcome.ok ? .done : .failed
                $0.message = outcome.message
                $0.results = outcome.results.map(\.file)
                if outcome.ok { $0.stageIndex = $0.stages.count }
            }
            self.lock.withLock { self.running = false }
            self.runNext()
        }
    }

    private func update(_ id: String, _ change: (inout Job) -> Void) {
        let job: Job? = lock.withLock {
            guard var job = jobs[id] else { return nil }
            change(&job)
            jobs[id] = job
            return job
        }
        if let job { save(job) }
        notify()
    }

    private func notify() {
        onChange?(allJobs)
    }

    static func safeName(_ name: String) -> Bool {
        !name.isEmpty && !name.contains("/") && !name.contains("..") && !name.hasPrefix(".")
    }

    // MARK: Persistence

    private var jobsFolder: URL { folder.appendingPathComponent("jobs") }

    private func save(_ job: Job) {
        let dir = jobsFolder.appendingPathComponent(job.id)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? JSONEncoder.link.encode(job).write(to: dir.appendingPathComponent("job.json"))
    }

    private func saveResultPaths(_ id: String) {
        let paths = lock.withLock { resultPaths[id] ?? [:] }
        try? JSONEncoder().encode(paths).write(to: jobsFolder.appendingPathComponent(id).appendingPathComponent("results.json"))
    }

    private func saveTokens() {
        let copy = lock.withLock { tokens }
        try? JSONEncoder().encode(copy).write(to: folder.appendingPathComponent("tokens.json"))
    }

    private func load() {
        if let data = try? Data(contentsOf: folder.appendingPathComponent("tokens.json")),
           let saved = try? JSONDecoder().decode([String: String].self, from: data) {
            tokens = saved
        }
        let ids = (try? FileManager.default.contentsOfDirectory(atPath: jobsFolder.path)) ?? []
        for id in ids {
            let dir = jobsFolder.appendingPathComponent(id)
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("job.json")),
                  var job = try? JSONDecoder.link.decode(Job.self, from: data) else { continue }
            // A job that was running when the Mac app quit did not finish.
            if job.status == .running {
                job.status = .failed
                job.message = "The Mac app was closed while this was analysed. Run it again."
            }
            jobs[id] = job
            if let paths = try? Data(contentsOf: dir.appendingPathComponent("results.json")),
               let saved = try? JSONDecoder().decode([String: String].self, from: paths) {
                resultPaths[id] = saved
            }
        }
    }

    /// Picks up queued jobs left from before (call once the runner is ready).
    public func resume() {
        runNext()
    }
}
