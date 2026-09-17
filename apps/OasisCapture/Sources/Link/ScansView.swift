import SwiftUI
import OasisLink

/// Recordings on the phone (send them to the Mac) and analyses on the Mac
/// (follow them, see the results).
struct ScansView: View {
    @ObservedObject var link = LinkStore.shared
    @State private var recordings: [URL] = []

    var body: some View {
        NavigationStack {
            List {
                if let upload = link.upload {
                    Section("Sending") { UploadRow(upload: upload) }
                }

                Section {
                    if link.paired == nil && !link.cloud.isAvailable {
                        Label("Pair with your Mac in the Mac tab, or sign in there, to analyse recordings.", systemImage: "laptopcomputer")
                            .foregroundStyle(.secondary)
                    } else if link.jobs.isEmpty && link.cloudJobs.isEmpty {
                        Text("Nothing analysed yet.").foregroundStyle(.secondary)
                    }
                    ForEach(link.jobs) { job in
                        NavigationLink(value: job.id) { JobRow(job: job) }
                    }
                    ForEach(link.cloudJobs) { job in
                        NavigationLink(value: job.id) { JobRow(job: job) }
                    }
                } header: {
                    Text(link.cloudJobs.isEmpty ? "On your Mac" : "On your Mac and in the cloud")
                }

                Section {
                    if recordings.isEmpty {
                        Text("Recordings you make appear here.").foregroundStyle(.secondary)
                    }
                    ForEach(recordings, id: \.self) { folder in
                        RecordingRow(folder: folder)
                    }
                } header: {
                    Text("On this phone")
                }
            }
            .navigationTitle("Scans")
            .navigationDestination(for: String.self) { id in
                JobDetail(jobID: id)
            }
            .refreshable { await link.refresh() }
            .onAppear {
                recordings = link.recordings()
                link.startBrowsing()
                link.startPolling()
            }
            .onDisappear { link.stopPolling() }
        }
    }
}

private struct UploadRow: View {
    let upload: LinkStore.Upload

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let error = upload.error {
                Label(error, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            } else {
                Text(upload.step).font(.subheadline)
                ProgressView(value: upload.progress)
                Text("\(Int(upload.progress * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct RecordingRow: View {
    @ObservedObject var link = LinkStore.shared
    let folder: URL
    @State private var name = ""
    @State private var asking = false

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.subheadline.weight(.medium))
                Text(link.sentJobID(folder) != nil ? "Sent to the Mac" : "Not sent yet")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button(link.sentJobID(folder) != nil ? "Send again" : "Send") {
                name = ""
                asking = true
            }
            .buttonStyle(.bordered)
            .disabled(!(link.canSendToMac || link.canSendThroughCloud)
                      || (link.upload != nil && link.upload?.error == nil && (link.upload?.progress ?? 0) < 1))
        }
        .alert("Name this room", isPresented: $asking) {
            TextField("Bedroom", text: $name)
            if link.canSendToMac {
                Button("Send to \(link.paired?.name ?? "Mac")") { send(.mac) }
            }
            if link.canSendThroughCloud {
                Button(link.canSendToMac ? "Send through the cloud" : "Send") { send(.cloud) }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(link.canSendToMac ? "Your Mac is on this Wi-Fi. The cloud works from anywhere; your Mac picks it up when it is on."
                 : "Goes through the cloud to whichever Mac is signed in to your account.")
        }
    }

    private func send(_ route: LinkStore.Route) {
        let room = name.trimmingCharacters(in: .whitespaces)
        Task { await link.send(recording: folder, name: room.isEmpty ? "Room" : room, via: route) }
    }

    private var title: String {
        folder.lastPathComponent.replacingOccurrences(of: "Capture-", with: "")
    }
}

struct JobRow: View {
    let job: Job

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(job.name).font(.headline)
                if job.cloud {
                    Image(systemName: "icloud").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: icon).foregroundStyle(color)
            }
            if job.status == .running || job.status == .queued || job.status == .receiving {
                ProgressView(value: job.progress)
            }
            Text(status)
                .font(.caption)
                .foregroundStyle(job.status == .failed ? .red : .secondary)
                .lineLimit(2)
        }
        .padding(.vertical, 3)
    }

    private var icon: String {
        switch job.status {
        case .receiving: return "arrow.up.circle"
        case .queued: return "clock"
        case .running: return "gearshape.2"
        case .done: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }

    private var color: Color {
        switch job.status {
        case .done: return .green
        case .failed: return .red
        default: return .blue
        }
    }

    private var status: String {
        switch job.status {
        case .receiving: return job.cloud ? "Being uploaded" : "Being sent to the Mac"
        case .queued: return job.cloud ? "Waiting for a Mac signed in to your account" : "Waiting its turn on the Mac"
        case .running: return "Step \(job.stageIndex + 1) of \(job.stages.count): \(job.message ?? job.currentStage ?? "")"
        case .done: return "Ready to view"
        case .failed: return job.message ?? "The analysis failed"
        }
    }
}

/// One analysis: progress while it runs, the room and the 3D view when done.
struct JobDetail: View {
    @ObservedObject var link = LinkStore.shared
    let jobID: String
    @State private var viewing = false

    private var job: Job? { link.job(jobID) }

    var body: some View {
        List {
            if let job {
                Section { JobRow(job: job) }

                if job.status == .running || job.status == .queued {
                    Section("Steps") {
                        ForEach(Array(job.stages.enumerated()), id: \.offset) { index, stage in
                            HStack {
                                Image(systemName: index < job.stageIndex ? "checkmark.circle.fill"
                                      : index == job.stageIndex && job.status == .running ? "circle.dotted" : "circle")
                                    .foregroundStyle(index < job.stageIndex ? .green : .secondary)
                                Text(stageName(stage))
                            }
                        }
                    }
                }

                if job.status == .done {
                    if let image = link.localResult(job, "room-render.png"), let ui = UIImage(contentsOfFile: image.path) {
                        Section("Room model") {
                            Image(uiImage: ui)
                                .resizable()
                                .scaledToFit()
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                        }
                    }
                    if let plan = link.localResult(job, "room-render-plan.png"), let ui = UIImage(contentsOfFile: plan.path) {
                        Section("From above") {
                            Image(uiImage: ui)
                                .resizable()
                                .scaledToFit()
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                        }
                    }
                    Section {
                        if splatURL(job) != nil {
                            Button {
                                viewing = true
                            } label: {
                                Label("View in 3D", systemImage: "cube.transparent")
                            }
                        } else {
                            HStack {
                                ProgressView()
                                Text(job.cloud ? "Fetching the results from the cloud…" : "Fetching the results from the Mac…").foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            } else {
                Text("This analysis is no longer on the Mac.").foregroundStyle(.secondary)
            }
        }
        .navigationTitle(job?.name ?? "Scan")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: job?.status) {
            if let job, job.status == .done { await link.fetchResults(job) }
        }
        .fullScreenCover(isPresented: $viewing) {
            if let job, let url = splatURL(job) {
                SplatScreen(splat: url, view: link.localResult(job, url.deletingPathExtension().lastPathComponent + ".view.json"))
            }
        }
    }

    private func splatURL(_ job: Job) -> URL? {
        link.localResult(job, "splat-filled.splat") ?? link.localResult(job, "splat.splat")
    }

    private func stageName(_ stage: String) -> String {
        switch stage {
        case "reconstruct": return "Finding where the phone was"
        case "densify": return "Measuring depth and recognising objects"
        case "shapes": return "Building the room model"
        case "splat": return "Training the 3D splat"
        default: return stage
        }
    }
}
