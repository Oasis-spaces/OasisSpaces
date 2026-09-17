import SwiftUI
import OasisLink

/// On the home page: pair phones, and follow the recordings they send.
struct PhoneCapturesSection: View {
    @Environment(StationService.self) private var station
    @State private var editingRepository = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                Label("Phone captures", systemImage: "iphone.gen3.radiowaves.left.and.right")
                    .font(.system(size: 17, weight: .semibold))
                Spacer()
                statusBadge
            }

            AccountBox(station: station)

            HStack(alignment: .top, spacing: 16) {
                pairingCard
                VStack(alignment: .leading, spacing: 6) {
                    Text("Open Oasis Capture on your phone, on the same Wi-Fi as this Mac. It finds this Mac by itself; enter the code once to pair. Recordings you send are analysed here, and the results show on both. Signed in on both, recordings also reach this Mac through the cloud from anywhere.")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 6) {
                        Text("Pipeline:")
                            .font(.system(size: 11))
                            .foregroundStyle(.tertiary)
                        if editingRepository {
                            @Bindable var station = station
                            TextField("OasisSpaces folder", text: $station.repositoryPath)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 11))
                                .frame(maxWidth: 320)
                                .onSubmit { editingRepository = false }
                        } else {
                            Text(station.repositoryPath)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(.secondary)
                            Button("Change") { editingRepository = true }
                                .buttonStyle(.link)
                                .font(.system(size: 11))
                        }
                    }
                }
            }

            if let problem = station.cloudProblem {
                Text(problem)
                    .font(.system(size: 11))
                    .foregroundStyle(.orange)
            }
            if station.jobs.isEmpty {
                Text("No recordings yet.")
                    .font(.system(size: 12))
                    .foregroundStyle(.tertiary)
            } else {
                VStack(spacing: 8) {
                    ForEach(station.jobs) { JobRow(job: $0) }
                }
            }
        }
        .padding(18)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
        .onAppear { station.start() }
    }

    @ViewBuilder
    private var statusBadge: some View {
        if let problem = station.problem {
            Label(problem, systemImage: "exclamationmark.triangle.fill")
                .font(.system(size: 11))
                .foregroundStyle(.orange)
        } else {
            Label(station.listening ? "Ready on this network" : "Starting…",
                  systemImage: station.listening ? "dot.radiowaves.left.and.right" : "hourglass")
                .font(.system(size: 11))
                .foregroundStyle(station.listening ? .green : .secondary)
        }
    }

    private var pairingCard: some View {
        VStack(spacing: 6) {
            Text("Pairing code")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
            Text(station.pairingCode.isEmpty ? "------" : spaced(station.pairingCode))
                .font(.system(size: 26, weight: .semibold, design: .monospaced))
                .textSelection(.enabled)
            HStack(spacing: 10) {
                Button("New code") { station.newPairingCode() }
                if !station.pairedDevices.isEmpty {
                    Button("Unpair all") { station.unpairAll() }
                }
            }
            .buttonStyle(.link)
            .font(.system(size: 11))
            Text(station.pairedDevices.isEmpty ? "No phone paired" : "Paired: \(station.pairedDevices.joined(separator: ", "))")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
        .padding(12)
        .frame(width: 200)
        .background(.background, in: RoundedRectangle(cornerRadius: 10))
    }

    private func spaced(_ code: String) -> String {
        code.prefix(3) + " " + code.suffix(3)
    }
}

private struct JobRow: View {
    @Environment(StationService.self) private var station
    let job: Job

    var body: some View {
        HStack(spacing: 12) {
            if let image = station.resultURL(job, "room-render.png"), let nsImage = NSImage(contentsOf: image) {
                Image(nsImage: nsImage)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 64, height: 44)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                RoundedRectangle(cornerRadius: 6)
                    .fill(.quaternary)
                    .frame(width: 64, height: 44)
                    .overlay(Image(systemName: icon).foregroundStyle(.secondary))
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(job.name).font(.system(size: 13, weight: .medium))
                    if job.cloud {
                        Image(systemName: "icloud")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                            .help("Sent through the cloud")
                    }
                    Text(job.capturedAt, style: .date)
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                }
                if job.status == .running || job.status == .queued || job.status == .receiving {
                    ProgressView(value: job.progress)
                        .frame(maxWidth: 360)
                }
                Text(statusText)
                    .font(.system(size: 11))
                    .foregroundStyle(job.status == .failed ? .red : .secondary)
                    .lineLimit(2)
            }
            Spacer()
            if job.status == .done {
                Button("Open 3D") { station.open(job) }
                Button { station.reveal(job) } label: { Image(systemName: "folder") }
                    .help("Show in Finder")
            }
            if (job.status == .failed || job.status == .done) && !job.cloud {
                Button { station.runAgain(job) } label: { Image(systemName: "arrow.clockwise") }
                    .help("Analyse again")
            }
        }
        .padding(10)
        .background(.background, in: RoundedRectangle(cornerRadius: 8))
    }

    private var icon: String {
        switch job.status {
        case .receiving: return "arrow.down.circle"
        case .queued: return "clock"
        case .running: return "gearshape.2"
        case .done: return "cube.transparent"
        case .failed: return "exclamationmark.triangle"
        }
    }

    private var statusText: String {
        switch job.status {
        case .receiving:
            return job.cloud ? "The phone is uploading it (\(job.filesReceived.count) of \(job.filesExpected.count) files)"
                : "Receiving from the phone (\(job.filesReceived.count) of \(job.filesExpected.count) files)"
        case .queued: return job.cloud ? "In the cloud, waiting for this Mac" : "Waiting for the analysis before it"
        case .running:
            let step = "Step \(job.stageIndex + 1) of \(job.stages.count)"
            return job.message.map { "\(step): \($0)" } ?? step
        case .done: return job.message.map { "Done. \($0)" } ?? "Done"
        case .failed: return job.message ?? "The analysis failed"
        }
    }
}

/// Sign in on the Mac so phones on the same account pair by themselves.
private struct AccountBox: View {
    let station: StationService
    @ObservedObject var account: AccountStore

    init(station: StationService) {
        self.station = station
        account = station.account
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            AccountView(store: account)
            Text(account.isSignedIn
                 ? "A phone signed in to this account pairs with this Mac without a code."
                 : "Sign in here and on the phone with the same account, and they pair without a code.")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .background(.background, in: RoundedRectangle(cornerRadius: 10))
        .onChange(of: account.session) { station.applyAccount() }
    }
}
