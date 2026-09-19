import SwiftUI
import WebKit
import OasisLink

/// A room as a mixed scene (a textured shell with the scanned furniture as
/// movable pieces), shown by the repository's scene viewer. The Mac's own
/// phone-link server hands the viewer and the scene's files to this web view,
/// the same way it hands them to the phone.
struct RoomSceneView: View {
    let address: URL

    var body: some View {
        RoomWebView(address: address)
            .frame(minWidth: 820, minHeight: 560)
            .background(Color.black)
    }
}

private struct RoomWebView: NSViewRepresentable {
    let address: URL

    func makeNSView(context: Context) -> WKWebView {
        let view = WKWebView(frame: .zero, configuration: WKWebViewConfiguration())
        view.setValue(false, forKey: "drawsBackground")
        view.load(URLRequest(url: address))
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        if view.url?.absoluteString != address.absoluteString { view.load(URLRequest(url: address)) }
    }
}

/// On the home page: every room in the repository that has a scene built.
struct RoomsSection: View {
    @Environment(StationService.self) private var station
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        let rooms = station.rooms
        if !rooms.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                Label("Rooms", systemImage: "square.split.bottomrightquarter")
                    .font(.system(size: 17, weight: .semibold))
                Text("Rooms rebuilt as a clean shell with the scanned furniture as pieces you can move, turn, hide or swap for a clean model.")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                VStack(spacing: 8) {
                    ForEach(rooms) { room in
                        HStack(spacing: 12) {
                            Image(systemName: "cube.transparent")
                                .font(.system(size: 18))
                                .foregroundStyle(.secondary)
                                .frame(width: 30)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(room.name).font(.system(size: 13, weight: .medium))
                                Text(room.detail)
                                    .font(.system(size: 11))
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }
                            Spacer()
                            Button("Open room") {
                                if let address = station.sceneAddress(folder: room.folder) { openWindow(id: "room", value: address) }
                            }
                            .disabled(!station.listening)
                        }
                        .padding(10)
                        .background(.background, in: RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
            .padding(18)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 14))
        }
    }
}

/// A built scene found in the repository's spaces.
struct RoomEntry: Identifiable {
    let folder: URL
    let name: String
    let detail: String
    var id: String { folder.path }

    /// Every spaces/<name>/scene/scene.json under the repository, newest first.
    static func all(in repository: URL) -> [RoomEntry] {
        let spaces = repository.appendingPathComponent("spaces")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: spaces.path)) ?? []
        struct Manifest: Decodable {
            struct Piece: Decodable { var label: String; var movable: Bool }
            struct Size: Decodable { var width: Double; var depth: Double }
            var room: Size
            var pieces: [Piece]
            var summary: String?
        }
        let found: [(Date, RoomEntry)] = names.compactMap { name in
            let folder = spaces.appendingPathComponent(name).appendingPathComponent("scene")
            let file = folder.appendingPathComponent("scene.json")
            guard let data = try? Data(contentsOf: file), let manifest = try? JSONDecoder().decode(Manifest.self, from: data) else { return nil }
            let furniture = manifest.pieces.filter(\.movable).map(\.label).joined(separator: ", ")
            let size = String(format: "%.1f × %.1f m", manifest.room.width, manifest.room.depth)
            let detail = manifest.summary ?? [size, furniture].filter { !$0.isEmpty }.joined(separator: " · ")
            let date = (try? file.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return (date, RoomEntry(folder: folder, name: name.replacingOccurrences(of: "-", with: " "), detail: detail))
        }
        return found.sorted { $0.0 > $1.0 }.map(\.1)
    }
}
