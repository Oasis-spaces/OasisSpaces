import AppKit
import Observation

struct SplatItem: Identifiable, Codable, Hashable {
    let id: UUID
    var path: String
    var addedAt: Date
    var splatCount: Int?
    var byteCount: Int64?

    var url: URL { URL(fileURLWithPath: path) }
    var fileName: String { url.lastPathComponent }
    var folderName: String { url.deletingLastPathComponent().lastPathComponent }
    var exists: Bool { FileManager.default.fileExists(atPath: path) }

    /// The pipeline names every splat splat.ply or splat-edited.splat, so its folder
    /// (the space's name) says which room it is.
    var title: String {
        let stem = url.deletingPathExtension().lastPathComponent
        if stem == "splat" { return folderName }
        if stem.hasPrefix("splat-") { return "\(folderName) · \(stem.dropFirst(6))" }
        return stem
    }
}

/// The splats the user has added, which of them are open as tabs, and which tab shows.
@MainActor
@Observable
final class Library {
    static let shared = Library(persistent: true)

    private(set) var items: [SplatItem] = []
    private(set) var openTabs: [UUID] = []
    /// The tab on screen; nil shows the library (the landing page).
    var selected: UUID?
    var importing = false
    /// Files that could not be added, shown on the landing page.
    var notice: String?
    private(set) var thumbnailRevision: [UUID: Int] = [:]

    private let persistent: Bool
    private let defaultsKey = "library.items.v1"
    let supportDirectory: URL

    init(persistent: Bool) {
        self.persistent = persistent
        supportDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Splat Viewer", isDirectory: true)
        try? FileManager.default.createDirectory(at: thumbnailDirectory, withIntermediateDirectories: true)
        if persistent, let data = UserDefaults.standard.data(forKey: defaultsKey),
           let saved = try? JSONDecoder().decode([SplatItem].self, from: data) {
            items = saved
        }
    }

    func item(_ id: UUID) -> SplatItem? { items.first { $0.id == id } }

    // MARK: Adding and removing

    /// Adds files (or folders of them) to the library and opens them as tabs.
    func add(_ urls: [URL], open shouldOpen: Bool = true) {
        var accepted: [UUID] = []
        var problems: [String] = []
        for dropped in urls {
            let files = SplatFile.expand(dropped)
            if files.isEmpty {
                problems.append("No Gaussian splats in \(dropped.lastPathComponent)")
            }
            for url in files {
                if let problem = SplatFile.problem(with: url) {
                    problems.append(problem)
                    continue
                }
                let path = url.standardizedFileURL.path
                if let existing = items.first(where: { $0.path == path }) {
                    accepted.append(existing.id)
                    continue
                }
                let item = SplatItem(id: UUID(), path: path, addedAt: Date(),
                                     splatCount: SplatFile.expectedCount(url),
                                     byteCount: SplatFile.byteCount(url))
                items.insert(item, at: 0)
                accepted.append(item.id)
            }
        }
        save()
        if problems.isEmpty {
            notice = nil
        } else {
            let shown = problems.prefix(3).joined(separator: "\n")
            notice = problems.count > 3 ? shown + "\n…and \(problems.count - 3) more" : shown
        }
        guard shouldOpen, let first = accepted.first else { return }
        for id in accepted { openTab(id, select: false) }
        selected = first
    }

    /// Takes a splat off the library (the file stays where it is).
    func remove(_ id: UUID) {
        close(id)
        items.removeAll { $0.id == id }
        try? FileManager.default.removeItem(at: thumbnailURL(for: id))
        save()
    }

    // MARK: Tabs

    func openTab(_ id: UUID, select: Bool = true) {
        guard item(id) != nil else { return }
        if !openTabs.contains(id) { openTabs.append(id) }
        if select { selected = id }
    }

    /// Closes a tab and shows its neighbour, or the library when it was the last one.
    func close(_ id: UUID) {
        guard let index = openTabs.firstIndex(of: id) else { return }
        openTabs.remove(at: index)
        if selected == id {
            selected = openTabs.isEmpty ? nil : openTabs[min(index, openTabs.count - 1)]
        }
    }

    @discardableResult
    func closeSelected() -> Bool {
        guard let selected else { return false }
        close(selected)
        return true
    }

    func showHome() { selected = nil }

    /// Tab by position among the open splats (0 = the first splat).
    func selectTab(at index: Int) {
        guard openTabs.indices.contains(index) else { return }
        selected = openTabs[index]
    }

    /// Cycles through the library and the open tabs.
    func selectAdjacent(_ delta: Int) {
        let order: [UUID?] = [nil] + openTabs.map { Optional($0) }
        let current = order.firstIndex(of: selected) ?? 0
        selected = order[((current + delta) % order.count + order.count) % order.count]
    }

    // MARK: What loading learns

    func recordLoaded(_ id: UUID, splatCount: Int) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        if items[index].splatCount != splatCount {
            items[index].splatCount = splatCount
            save()
        }
    }

    var thumbnailDirectory: URL { supportDirectory.appendingPathComponent("Thumbnails", isDirectory: true) }

    func thumbnailURL(for id: UUID) -> URL {
        thumbnailDirectory.appendingPathComponent("\(id.uuidString).png")
    }

    func thumbnailSaved(_ id: UUID) {
        thumbnailRevision[id, default: 0] += 1
    }

    func revealInFinder(_ id: UUID) {
        guard let item = item(id) else { return }
        NSWorkspace.shared.activateFileViewerSelecting([item.url])
    }

    private func save() {
        guard persistent, let data = try? JSONEncoder().encode(items) else { return }
        UserDefaults.standard.set(data, forKey: defaultsKey)
    }

    /// A library that is never saved, for SelfTest.
    static func scratch(with urls: [URL]) -> Library {
        let library = Library(persistent: false)
        library.add(urls, open: false)
        return library
    }
}
