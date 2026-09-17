import Foundation
import Network

/// A request's head; the body is read according to what the router decides.
public struct HTTPRequest: Sendable {
    public var method: String
    public var path: String
    public var query: [String: String]
    /// Header names in lower case.
    public var headers: [String: String]

    public var bearerToken: String? {
        guard let value = headers["authorization"], value.lowercased().hasPrefix("bearer ") else { return nil }
        return String(value.dropFirst(7)).trimmingCharacters(in: .whitespaces)
    }

    public var contentLength: Int64 { Int64(headers["content-length"] ?? "") ?? 0 }

    /// Path segments: "/jobs/abc/files/video.mov" -> ["jobs", "abc", "files", "video.mov"].
    public var segments: [String] {
        path.split(separator: "/").map { String($0).removingPercentEncoding ?? String($0) }
    }
}

public struct HTTPResponse: Sendable {
    public enum Body: Sendable {
        case none
        case data(Data)
        case file(URL)
    }

    public var status: Int
    public var headers: [String: String]
    public var body: Body

    public init(status: Int, headers: [String: String] = [:], body: Body = .none) {
        self.status = status
        self.headers = headers
        self.body = body
    }

    public static func json<T: Encodable>(_ value: T, status: Int = 200) -> HTTPResponse {
        let data = (try? JSONEncoder.link.encode(value)) ?? Data("{}".utf8)
        return HTTPResponse(status: status, headers: ["Content-Type": "application/json"], body: .data(data))
    }

    public static func error(_ status: Int, _ message: String) -> HTTPResponse {
        json(["error": message], status: status)
    }
}

/// What to do with a request once its head is known.
public enum RouteDecision: Sendable {
    /// Answer without reading a body.
    case respond(HTTPResponse)
    /// Read a small body (up to maxBytes) into memory, then answer.
    case body(maxBytes: Int, @Sendable (Data) -> HTTPResponse)
    /// Stream the body to a file, then answer (the Bool says whether all of it arrived).
    case upload(to: URL, @Sendable (Bool) -> HTTPResponse)
}

/// A small HTTP/1.1 server on Network.framework: one request per connection,
/// uploads streamed to disk, files streamed back in chunks. Enough for a phone
/// and a Mac on the same network; not a general web server.
public final class HTTPServer: @unchecked Sendable {
    public typealias Router = @Sendable (HTTPRequest) -> RouteDecision

    private let listener: NWListener
    private let router: Router
    private let queue = DispatchQueue(label: "oasislink.server")
    private static let headerLimit = 64 * 1024
    private static let chunk = 1 << 20

    /// `service` advertises the server over Bonjour with a TXT record.
    public init(port: UInt16, service: (name: String, type: String, txt: [String: String])? = nil,
                router: @escaping Router) throws {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        listener = try NWListener(using: parameters, on: NWEndpoint.Port(rawValue: port) ?? .any)
        if let service {
            listener.service = NWListener.Service(name: service.name, type: service.type, domain: nil,
                                                  txtRecord: NWTXTRecord(service.txt))
        }
        self.router = router
    }

    /// The port it listens on, once ready.
    public var port: UInt16? { listener.port?.rawValue }

    public func start(onState: (@Sendable (NWListener.State) -> Void)? = nil) {
        listener.stateUpdateHandler = { onState?($0) }
        listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
        listener.start(queue: queue)
    }

    public func stop() {
        listener.cancel()
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        readHead(connection, buffer: Data())
    }

    private func readHead(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let end = buffer.range(of: Data("\r\n\r\n".utf8)) {
                let head = buffer.subdata(in: 0..<end.lowerBound)
                let rest = buffer.subdata(in: end.upperBound..<buffer.count)
                guard let request = Self.parse(head) else {
                    self.send(.error(400, "bad request"), on: connection)
                    return
                }
                self.handle(request, rest: rest, on: connection)
            } else if buffer.count > Self.headerLimit || isComplete || error != nil {
                connection.cancel()
            } else {
                self.readHead(connection, buffer: buffer)
            }
        }
    }

    static func parse(_ head: Data) -> HTTPRequest? {
        guard let text = String(data: head, encoding: .utf8) else { return nil }
        var lines = text.components(separatedBy: "\r\n")
        let requestLine = lines.removeFirst().split(separator: " ")
        guard requestLine.count >= 2 else { return nil }
        var headers: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        let target = String(requestLine[1])
        let components = URLComponents(string: target)
        var query: [String: String] = [:]
        for item in components?.queryItems ?? [] { query[item.name] = item.value ?? "" }
        return HTTPRequest(method: String(requestLine[0]), path: components?.path ?? target, query: query, headers: headers)
    }

    private func handle(_ request: HTTPRequest, rest: Data, on connection: NWConnection) {
        switch router(request) {
        case .respond(let response):
            send(response, on: connection)
        case .body(let maxBytes, let answer):
            let length = Int(request.contentLength)
            guard length <= maxBytes else {
                send(.error(413, "body too large"), on: connection)
                return
            }
            readBody(connection, have: rest, length: length) { [weak self] data in
                guard let data else { connection.cancel(); return }
                self?.send(answer(data), on: connection)
            }
        case .upload(let url, let answer):
            let length = request.contentLength
            FileManager.default.createFile(atPath: url.path, contents: nil)
            guard let handle = try? FileHandle(forWritingTo: url) else {
                send(.error(500, "cannot write \(url.lastPathComponent)"), on: connection)
                return
            }
            let first = rest.prefix(Int(min(Int64(rest.count), length)))
            handle.write(first)
            stream(connection, to: handle, remaining: length - Int64(first.count)) { [weak self] complete in
                try? handle.close()
                self?.send(answer(complete), on: connection)
            }
        }
    }

    private func readBody(_ connection: NWConnection, have: Data, length: Int, done: @escaping (Data?) -> Void) {
        if have.count >= length {
            done(have.prefix(length))
            return
        }
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            var have = have
            if let data { have.append(data) }
            if have.count >= length {
                done(have.prefix(length))
            } else if isComplete || error != nil {
                done(nil)
            } else {
                self?.readBody(connection, have: have, length: length, done: done)
            }
        }
    }

    private func stream(_ connection: NWConnection, to handle: FileHandle, remaining: Int64,
                        done: @escaping (Bool) -> Void) {
        if remaining <= 0 {
            done(true)
            return
        }
        connection.receive(minimumIncompleteLength: 1, maximumLength: Self.chunk) { [weak self] data, _, isComplete, error in
            var remaining = remaining
            if let data, !data.isEmpty {
                let part = data.prefix(Int(min(Int64(data.count), remaining)))
                handle.write(part)
                remaining -= Int64(part.count)
            }
            if remaining <= 0 {
                done(true)
            } else if isComplete || error != nil {
                done(false)
            } else {
                self?.stream(connection, to: handle, remaining: remaining, done: done)
            }
        }
    }

    private func send(_ response: HTTPResponse, on connection: NWConnection) {
        var headers = response.headers
        headers["Connection"] = "close"
        var fileHandle: FileHandle?
        var length: Int64 = 0
        switch response.body {
        case .none: length = 0
        case .data(let data): length = Int64(data.count)
        case .file(let url):
            fileHandle = try? FileHandle(forReadingFrom: url)
            length = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
            if fileHandle == nil {
                send(.error(404, "not found"), on: connection)
                return
            }
            headers["Content-Type"] = headers["Content-Type"] ?? "application/octet-stream"
        }
        headers["Content-Length"] = String(length)
        var head = "HTTP/1.1 \(response.status) \(Self.reason(response.status))\r\n"
        for (k, v) in headers { head += "\(k): \(v)\r\n" }
        head += "\r\n"
        var first = Data(head.utf8)
        if case .data(let data) = response.body { first.append(data) }
        connection.send(content: first, completion: .contentProcessed { [weak self] error in
            guard error == nil, let fileHandle else {
                connection.cancel()
                return
            }
            self?.sendFile(fileHandle, on: connection)
        })
    }

    private func sendFile(_ handle: FileHandle, on connection: NWConnection) {
        let data = handle.readData(ofLength: Self.chunk)
        if data.isEmpty {
            try? handle.close()
            connection.send(content: nil, isComplete: true, completion: .contentProcessed { _ in connection.cancel() })
            return
        }
        connection.send(content: data, completion: .contentProcessed { [weak self] error in
            if error != nil {
                try? handle.close()
                connection.cancel()
            } else {
                self?.sendFile(handle, on: connection)
            }
        })
    }

    private static func reason(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 201: return "Created"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 403: return "Forbidden"
        case 404: return "Not Found"
        case 409: return "Conflict"
        case 413: return "Payload Too Large"
        default: return status < 400 ? "OK" : "Error"
        }
    }
}
