import Foundation

/// Prints, and appends to Documents/oasis-capture.log so the log can be read
/// off the phone (the app allows file sharing) when no console is attached.
enum AppLog {
    private static let queue = DispatchQueue(label: "capture.log", qos: .utility)
    private static let url: URL = {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documents.appendingPathComponent("oasis-capture.log")
    }()
    private static let stamp: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    static func write(_ message: String) {
        print("Oasis Capture: \(message)")
        let line = "\(stamp.string(from: Date())) \(message)\n"
        queue.async {
            guard let data = line.data(using: .utf8) else { return }
            if let handle = try? FileHandle(forWritingTo: url) {
                _ = try? handle.seekToEnd()
                try? handle.write(contentsOf: data)
                try? handle.close()
            } else {
                try? data.write(to: url)
            }
        }
    }

    /// Starts a fresh log (kept short: the last launch only).
    static func begin() {
        queue.async { try? FileManager.default.removeItem(at: url) }
        write("launch \(Date())")
    }
}
