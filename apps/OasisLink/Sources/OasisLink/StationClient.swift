import Foundation

public struct LinkError: Error, LocalizedError, Sendable {
    public var status: Int
    public var message: String
    public var errorDescription: String? { message }
    public init(status: Int, message: String) {
        self.status = status
        self.message = message
    }
}

/// Talks to one Mac station. Every call but info and pair needs the token
/// the Mac gave when paired.
public final class StationClient: NSObject, @unchecked Sendable {
    public let baseURL: URL
    public var token: String?
    private let session: URLSession

    /// e.g. http://Sanskars-MacBook-Air.local:8765
    public init(baseURL: URL, token: String? = nil) {
        self.baseURL = baseURL
        self.token = token
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 60 * 60 * 2   // a large video on slow Wi-Fi
        configuration.waitsForConnectivity = false
        session = URLSession(configuration: configuration)
    }

    public func info() async throws -> StationInfo {
        try await get("info")
    }

    public func pair(code: String, device: String) async throws -> PairResponse {
        let response: PairResponse = try await send("POST", "pair", body: PairRequest(code: code, device: device))
        token = response.token
        return response
    }

    /// Pairs with a Mac signed in to the same account, without a code.
    public func pair(account: AccountSession, device: String) async throws -> PairResponse {
        let response: PairResponse = try await send("POST", "pair/account",
                                                    body: AccountPairRequest(accessToken: account.accessToken, device: device))
        token = response.token
        return response
    }

    public func jobs() async throws -> [Job] {
        try await get("jobs")
    }

    public func job(_ id: String) async throws -> Job {
        try await get("jobs/\(id)")
    }

    public func createJob(_ job: NewJob) async throws -> Job {
        try await send("POST", "jobs", body: job)
    }

    /// Streams a file to the Mac; `progress` gets 0...1 on an arbitrary queue.
    public func upload(file: URL, to jobID: String, as name: String,
                       progress: @escaping @Sendable (Double) -> Void) async throws {
        var request = URLRequest(url: url("jobs/\(jobID)/files/\(name)"))
        request.httpMethod = "PUT"
        authorize(&request)
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        let delegate = UploadProgress(progress)
        let (data, response) = try await session.upload(for: request, fromFile: file, delegate: delegate)
        try check(response, data)
    }

    public func startJob(_ id: String) async throws -> Job {
        try await send("POST", "jobs/\(id)/start", body: [String: String]())
    }

    /// Downloads a result into `folder`; returns the file.
    public func download(_ result: ResultFile, of jobID: String, into folder: URL) async throws -> URL {
        var request = URLRequest(url: url("jobs/\(jobID)/results/\(result.name)"))
        authorize(&request)
        let (temporary, response) = try await session.download(for: request)
        try check(response, nil)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let target = folder.appendingPathComponent(result.name)
        try? FileManager.default.removeItem(at: target)
        try FileManager.default.moveItem(at: temporary, to: target)
        return target
    }

    // MARK: The cloud relay's routes (the same client, a few more calls)

    public struct SignedUpload: Decodable, Sendable {
        public var url: URL
        public var partBytes: Int?
    }

    /// Where to PUT one part of one input file.
    public func uploadURL(job: String, file: String, part: Int) async throws -> SignedUpload {
        try await send("POST", "jobs/\(job)/upload", body: ["file": file, "part": String(part)])
    }

    /// All parts of one input file are up.
    public func received(job: String, file: String, parts: Int) async throws -> Job {
        try await send("POST", "jobs/\(job)/received", body: ["file": file, "parts": String(parts)])
    }

    /// Jobs in one state (the relay filters; a Mac station returns everything).
    public func jobs(status: JobStatus) async throws -> [Job] {
        try await get("jobs?status=\(status.rawValue)")
    }

    /// Takes a queued cloud job for this Mac; fails with 409 when another Mac was first.
    public func claim(job: String, station: String) async throws -> Job {
        try await send("POST", "jobs/\(job)/claim", body: ["station": station])
    }

    /// Signed downloads of every part of one input file, in order.
    public func downloadURLs(job: String, file: String) async throws -> [URL] {
        struct Reply: Decodable { var urls: [URL] }
        let reply: Reply = try await send("POST", "jobs/\(job)/download", body: ["file": file])
        return reply.urls
    }

    public struct Progress: Encodable, Sendable {
        public var stageIndex: Int?
        public var message: String?
        public var status: JobStatus?
        public var results: [ResultFile]?
        public init(stageIndex: Int? = nil, message: String? = nil, status: JobStatus? = nil, results: [ResultFile]? = nil) {
            self.stageIndex = stageIndex; self.message = message; self.status = status; self.results = results
        }
    }

    /// Progress, outcome and results of a cloud job (from the Mac).
    public func report(job: String, _ progress: Progress) async throws -> Job {
        try await send("PATCH", "jobs/\(job)", body: progress)
    }

    /// Where to PUT one result file.
    public func resultUploadURL(job: String, name: String) async throws -> URL {
        struct Reply: Decodable { var url: URL }
        let reply: Reply = try await send("POST", "jobs/\(job)/results/upload", body: ["name": name])
        return reply.url
    }

    /// The Mac has the capture: the relay frees its storage.
    public func deleteInputs(job: String) async throws {
        let _: [String: Int] = try await send("POST", "jobs/\(job)/inputs/delete", body: [String: String]())
    }

    public func deleteJob(_ id: String) async throws {
        var request = URLRequest(url: url("jobs/\(id)"))
        request.httpMethod = "DELETE"
        authorize(&request)
        let (data, response) = try await session.data(for: request)
        try check(response, data)
    }

    /// PUTs bytes to a signed storage URL (no token: the URL carries its own).
    ///
    /// Sent at full speed first. Some networks corrupt an upload that leaves
    /// fast (the connection dies with a TLS error a few tens of kilobytes in,
    /// every time, while downloads are fine); the same bytes fed slowly arrive
    /// intact. So a connection-level failure is answered with one more try at
    /// `Self.pacedBytesPerSecond`.
    public func put(_ data: Data, to signed: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        do {
            var request = URLRequest(url: signed)
            request.httpMethod = "PUT"
            request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
            let (body, response) = try await session.upload(for: request, from: data, delegate: UploadProgress(progress))
            try check(response, body)
        } catch let error as URLError where error.code != .cancelled && error.code != .notConnectedToInternet {
            try await put(data, to: signed, bytesPerSecond: Self.pacedBytesPerSecond, progress: progress)
        }
    }

    /// The pace of the second try of an upload, bytes a second.
    public static let pacedBytesPerSecond = 150_000

    /// PUTs bytes no faster than `bytesPerSecond`: the body is a stream this
    /// side fills a little at a time, so the connection cannot send ahead of it.
    public func put(_ data: Data, to signed: URL, bytesPerSecond: Int,
                    progress: @escaping @Sendable (Double) -> Void) async throws {
        var request = URLRequest(url: signed)
        request.httpMethod = "PUT"
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        request.setValue(String(data.count), forHTTPHeaderField: "Content-Length")
        var input: InputStream?
        var output: OutputStream?
        Stream.getBoundStreams(withBufferSize: 8192, inputStream: &input, outputStream: &output)
        guard let input, let output else { throw LinkError(status: 0, message: "could not open the upload stream") }
        request.httpBodyStream = input
        let feeder = PacedFeeder(data: data, output: output, bytesPerSecond: max(1_000, bytesPerSecond), progress: progress)
        feeder.start()
        defer { feeder.cancel() }
        let (body, response) = try await session.data(for: request)
        try check(response, body)
    }

    /// Downloads a signed storage URL and appends it to `file` (parts in order make the file).
    public func fetch(_ signed: URL, appendingTo file: URL) async throws {
        let (temporary, response) = try await session.download(from: signed)
        try check(response, nil)
        if !FileManager.default.fileExists(atPath: file.path) {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.moveItem(at: temporary, to: file)
            return
        }
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        try handle.seekToEnd()
        let source = try FileHandle(forReadingFrom: temporary)
        defer { try? source.close(); try? FileManager.default.removeItem(at: temporary) }
        while let chunk = try source.read(upToCount: 8 << 20), !chunk.isEmpty {
            try handle.write(contentsOf: chunk)
        }
    }

    // MARK: -

    private func url(_ path: String) -> URL {
        URL(string: path, relativeTo: baseURL.appendingPathComponent("/"))!.absoluteURL
    }

    private func authorize(_ request: inout URLRequest) {
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        var request = URLRequest(url: url(path))
        authorize(&request)
        let (data, response) = try await session.data(for: request)
        try check(response, data)
        return try JSONDecoder.link.decode(T.self, from: data)
    }

    private func send<T: Decodable, B: Encodable>(_ method: String, _ path: String, body: B) async throws -> T {
        var request = URLRequest(url: url(path))
        request.httpMethod = method
        authorize(&request)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder.link.encode(body)
        let (data, response) = try await session.data(for: request)
        try check(response, data)
        return try JSONDecoder.link.decode(T.self, from: data)
    }

    private func check(_ response: URLResponse, _ data: Data?) throws {
        guard let http = response as? HTTPURLResponse else { throw LinkError(status: 0, message: "no response") }
        guard (200..<300).contains(http.statusCode) else {
            // A Mac station answers {"error": ...}; the relay {"detail": ...}.
            let fields = data.flatMap { try? JSONDecoder().decode([String: String].self, from: $0) }
            let message = fields?["error"] ?? fields?["detail"]
            throw LinkError(status: http.statusCode, message: message ?? "The server answered \(http.statusCode)")
        }
    }
}

/// Writes a body into a bound stream at a fixed pace, on its own thread (a
/// write to a full stream blocks, which a Swift task must not do).
private final class PacedFeeder: @unchecked Sendable {
    private let data: Data
    private let output: OutputStream
    private let bytesPerSecond: Int
    private let progress: @Sendable (Double) -> Void
    private let lock = NSLock()
    private var cancelled = false

    init(data: Data, output: OutputStream, bytesPerSecond: Int, progress: @escaping @Sendable (Double) -> Void) {
        self.data = data
        self.output = output
        self.bytesPerSecond = bytesPerSecond
        self.progress = progress
    }

    func cancel() {
        lock.withLock { cancelled = true }
    }

    func start() {
        Thread.detachNewThread { [self] in
            output.open()
            defer { output.close() }
            let piece = 2_000
            let pause = Double(piece) / Double(bytesPerSecond)
            var offset = 0
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.bindMemory(to: UInt8.self).baseAddress else { return }
                while offset < data.count {
                    if lock.withLock({ cancelled }) { return }
                    let wrote = output.write(base + offset, maxLength: min(piece, data.count - offset))
                    if wrote <= 0 { return }   // the request ended (or failed) before the body did
                    offset += wrote
                    progress(Double(offset) / Double(data.count))
                    Thread.sleep(forTimeInterval: pause * Double(wrote) / Double(piece))
                }
            }
        }
    }
}

private final class UploadProgress: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    let report: @Sendable (Double) -> Void
    init(_ report: @escaping @Sendable (Double) -> Void) { self.report = report }

    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        guard totalBytesExpectedToSend > 0 else { return }
        report(Double(totalBytesSent) / Double(totalBytesExpectedToSend))
    }
}
