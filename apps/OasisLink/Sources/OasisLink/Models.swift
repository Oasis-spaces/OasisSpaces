import Foundation

public enum Link {
    /// Bonjour service type the Mac advertises.
    public static let serviceType = "_oasisstation._tcp"
    public static let defaultPort: UInt16 = 8765
    public static let version = 1
    /// The files a recording is made of, in upload order (video last: it is the big one).
    public static let captureFiles = ["capture.json", "frames.jsonl", "video.mov"]
}

/// What GET /info returns.
public struct StationInfo: Codable, Sendable, Equatable {
    public var id: String
    public var name: String
    public var version: Int
    /// AccountSession.tag of the account the Mac is signed in to, if any.
    public var owner: String?
    public init(id: String, name: String, version: Int = Link.version, owner: String? = nil) {
        self.id = id
        self.name = name
        self.version = version
        self.owner = owner
    }
}

public struct PairRequest: Codable, Sendable {
    public var code: String
    public var device: String
    public init(code: String, device: String) {
        self.code = code
        self.device = device
    }
}

public struct PairResponse: Codable, Sendable {
    public var token: String
    public var station: StationInfo
    public init(token: String, station: StationInfo) {
        self.token = token
        self.station = station
    }
}

public struct NewJob: Codable, Sendable {
    public var name: String
    public var files: [String]
    public var capturedAt: Date
    public init(name: String, files: [String], capturedAt: Date) {
        self.name = name
        self.files = files
        self.capturedAt = capturedAt
    }
}

public enum JobStatus: String, Codable, Sendable {
    /// Created; files still arriving from the phone.
    case receiving
    /// All files in; waiting for its turn.
    case queued
    case running
    case done
    case failed
}

public struct ResultFile: Codable, Sendable, Equatable, Hashable {
    public enum Kind: String, Codable, Sendable { case splat, view, image, report }
    public var name: String
    public var kind: Kind
    public var bytes: Int64
    public init(name: String, kind: Kind, bytes: Int64) {
        self.name = name
        self.kind = kind
        self.bytes = bytes
    }
}

public struct Job: Codable, Sendable, Identifiable, Equatable {
    public var id: String
    public var name: String
    public var createdAt: Date
    public var capturedAt: Date
    public var status: JobStatus
    /// Pipeline stages in order, and how far it got.
    public var stages: [String]
    public var stageIndex: Int
    public var message: String?
    public var filesExpected: [String]
    public var filesReceived: [String]
    public var results: [ResultFile]

    public init(id: String, name: String, createdAt: Date, capturedAt: Date, status: JobStatus,
                stages: [String], stageIndex: Int = 0, message: String? = nil,
                filesExpected: [String], filesReceived: [String] = [], results: [ResultFile] = []) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.capturedAt = capturedAt
        self.status = status
        self.stages = stages
        self.stageIndex = stageIndex
        self.message = message
        self.filesExpected = filesExpected
        self.filesReceived = filesReceived
        self.results = results
    }

    /// 0...1 over the pipeline's stages.
    public var progress: Double {
        switch status {
        case .done: return 1
        case .receiving, .queued: return 0
        case .running, .failed: return stages.isEmpty ? 0 : Double(stageIndex) / Double(stages.count)
        }
    }

    public var currentStage: String? {
        status == .running && stageIndex < stages.count ? stages[stageIndex] : nil
    }
}

public extension JSONEncoder {
    static let link: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        return e
    }()
}

public extension JSONDecoder {
    static let link: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}
