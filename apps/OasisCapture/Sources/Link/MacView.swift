import SwiftUI
import OasisLink

/// Pairing with the Mac that analyses recordings.
struct MacView: View {
    @ObservedObject var link = LinkStore.shared
    @State private var choosing: DiscoveredStation?
    @State private var code = ""
    @State private var pairing = false

    var body: some View {
        NavigationStack {
            List {
                if let mac = link.paired {
                    Section {
                        HStack(spacing: 14) {
                            Image(systemName: "laptopcomputer")
                                .font(.largeTitle)
                                .foregroundStyle(.green)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(mac.name).font(.headline)
                                Label(link.reachable ? "Connected" : "Not reachable on this network",
                                      systemImage: link.reachable ? "wifi" : "wifi.exclamationmark")
                                    .font(.subheadline)
                                    .foregroundStyle(link.reachable ? .green : .orange)
                            }
                        }
                        .padding(.vertical, 6)
                        if let error = link.lastError {
                            Text(error).foregroundStyle(.orange).font(.footnote)
                        }
                        Button("Unpair", role: .destructive) { link.unpair() }
                    } header: {
                        Text("Your Mac")
                    } footer: {
                        Text("Recordings you send are analysed on this Mac. Your phone and the Mac need to be on the same Wi-Fi.")
                    }
                }

                Section {
                    if link.discovered.isEmpty {
                        HStack(spacing: 10) {
                            ProgressView()
                            Text("Looking for Macs on this Wi-Fi…").foregroundStyle(.secondary)
                        }
                    }
                    ForEach(link.discovered) { station in
                        Button {
                            code = ""
                            choosing = station
                        } label: {
                            HStack {
                                Label(station.name, systemImage: "laptopcomputer")
                                Spacer()
                                if link.paired?.id == station.id {
                                    Text("Paired").foregroundStyle(.green)
                                } else {
                                    Text("Pair").foregroundStyle(.blue)
                                }
                            }
                        }
                        .foregroundStyle(.primary)
                    }
                } header: {
                    Text(link.paired == nil ? "Pair with your Mac" : "Macs on this network")
                } footer: {
                    Text("On the Mac, open Splat Viewer. Its home page shows a pairing code under Phone captures.")
                }
            }
            .navigationTitle("Mac")
            .onAppear {
                link.startBrowsing()
                Task { await link.refresh() }
            }
            .sheet(item: $choosing) { station in
                NavigationStack {
                    Form {
                        Section {
                            TextField("6-digit code", text: $code)
                                .keyboardType(.numberPad)
                                .font(.system(size: 30, weight: .semibold, design: .monospaced))
                                .multilineTextAlignment(.center)
                        } footer: {
                            Text("Enter the code shown in Splat Viewer on \(station.name).")
                        }
                        if let error = link.lastError, !pairing {
                            Text(error).foregroundStyle(.red)
                        }
                    }
                    .navigationTitle("Pair")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) { Button("Cancel") { choosing = nil } }
                        ToolbarItem(placement: .confirmationAction) {
                            Button {
                                pairing = true
                                Task {
                                    let ok = await link.pair(with: station, code: code)
                                    pairing = false
                                    if ok { choosing = nil }
                                }
                            } label: {
                                if pairing { ProgressView() } else { Text("Pair") }
                            }
                            .disabled(code.filter(\.isNumber).count != 6 || pairing)
                        }
                    }
                }
                .presentationDetents([.medium])
            }
        }
    }
}
