import SwiftUI
import CaptureRules

/// After recording: how it went, what to watch next time, and where the files are.
struct ReviewView: View {
    let result: CaptureResult
    let config: RuleConfig
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 6) {
                        Label(verdict.title, systemImage: verdict.icon)
                            .font(.title2.bold())
                            .foregroundStyle(verdict.color)
                        Text(verdict.detail).foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 6)
                }

                Section("This recording") {
                    Stat(label: "Length", value: duration)
                    Stat(label: "Walked", value: String(format: "%.1f m", result.summary.pathLength))
                    Stat(label: "Walls covered", value: "\(Int((result.summary.coverage * 100).rounded()))%",
                         good: result.summary.coverage >= config.coverageGoal)
                    Stat(label: "Floor shown", value: result.summary.floorSeen ? "Yes" : "No",
                         good: result.summary.floorSeen)
                    Stat(label: "Ended from start", value: String(format: "%.1f m", result.summary.endDistanceFromStart),
                         good: result.summary.endDistanceFromStart <= config.loopCloseDistance)
                }

                if !result.advice.isEmpty {
                    Section("Before you finish") {
                        ForEach(result.advice, id: \.self) { rule in
                            Label(config.message(rule), systemImage: "lightbulb")
                        }
                    }
                }

                if !warnings.isEmpty {
                    Section("Moments to watch") {
                        ForEach(warnings, id: \.rule) { item in
                            HStack {
                                Text(config.message(item.rule))
                                Spacer()
                                Text(String(format: "%.0f s", item.seconds)).foregroundStyle(.secondary)
                            }
                        }
                    }
                }

                if !result.objectsSeen.isEmpty {
                    Section("Recognised in the room") {
                        Text(result.objectsSeen.joined(separator: ", "))
                    }
                }

                if let folder = result.folder {
                    SendSection(folder: folder)
                }

                Section("Files") {
                    if let folder = result.folder {
                        Text("Saved in Files › On My iPhone › Oasis Capture › Captures › \(folder.lastPathComponent)")
                            .font(.footnote)
                        ShareLink(items: shareItems(folder)) {
                            Label("Share video and camera track", systemImage: "square.and.arrow.up")
                        }
                    } else {
                        Text("The video could not be saved.").foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("Recording saved")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var duration: String {
        let s = Int(result.summary.duration)
        return String(format: "%d:%02d", s / 60, s % 60)
    }

    private var warnings: [(rule: Rule, seconds: Double)] {
        result.summary.secondsByRule
            .compactMap { key, seconds in
                guard let rule = Rule(rawValue: key), rule.severity >= .warn, seconds >= 1 else { return nil }
                return (rule, seconds)
            }
            .sorted { $0.seconds > $1.seconds }
    }

    private var verdict: (title: String, detail: String, icon: String, color: Color) {
        if result.advice.isEmpty && warnings.allSatisfy({ $0.seconds < 5 }) {
            return ("Great recording", "Everything the 3D model needs is in it.", "checkmark.seal.fill", .green)
        }
        if result.advice.contains(.tooShort) || result.summary.coverage < 0.5 {
            return ("Part of the room is missing", "You can record again, or keep this and add the parts you missed.",
                    "exclamationmark.triangle.fill", .orange)
        }
        return ("Good recording", "A few things below would make the next one better.", "checkmark.circle.fill", .blue)
    }

    private func shareItems(_ folder: URL) -> [URL] {
        ["video.mov", "frames.jsonl", "capture.json"]
            .map { folder.appendingPathComponent($0) }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
    }
}

private struct Stat: View {
    let label: String
    let value: String
    var good: Bool? = nil

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary)
            if let good {
                Image(systemName: good ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                    .foregroundStyle(good ? .green : .orange)
            }
        }
    }
}

/// Name the room and send the recording to the paired Mac for analysis.
private struct SendSection: View {
    @ObservedObject var link = LinkStore.shared
    let folder: URL
    @State private var name = ""

    var body: some View {
        Section {
            if link.paired == nil {
                Label("Pair with your Mac in the Mac tab to turn this into a 3D room.", systemImage: "laptopcomputer")
                    .foregroundStyle(.secondary)
            } else if let upload = link.upload, upload.recording == folder {
                if let error = upload.error {
                    Label(error, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                } else {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(upload.step)
                        ProgressView(value: upload.progress)
                    }
                }
            } else {
                TextField("Room name, e.g. Bedroom", text: $name)
                Button {
                    let room = name.trimmingCharacters(in: .whitespaces)
                    Task { await link.send(recording: folder, name: room.isEmpty ? "Room" : room) }
                } label: {
                    Label("Send to \(link.paired?.name ?? "Mac")", systemImage: "arrow.up.forward.app")
                }
            }
        } header: {
            Text("Make the 3D room")
        } footer: {
            Text("The Mac analyses it. Follow it and view the result in the Scans tab.")
        }
    }
}
