import Foundation

public struct LinkError: Error, LocalizedError, Sendable {
    public var status: Int
    public var message: String
    public var errorDescription: String? { message }
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
            let message = data.flatMap { try? JSONDecoder().decode([String: String].self, from: $0)["error"] }
            throw LinkError(status: http.statusCode, message: message ?? "The Mac answered \(http.statusCode)")
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
