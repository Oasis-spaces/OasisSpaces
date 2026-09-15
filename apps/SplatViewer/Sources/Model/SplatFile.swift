import Foundation
import UniformTypeIdentifiers

/// Which files can be opened, and why a file cannot.
enum SplatFile {
    static let extensions: Set<String> = ["ply", "splat", "spz"]

    static var contentTypes: [UTType] {
        extensions.sorted().compactMap { UTType(filenameExtension: $0) } + [.folder]
    }

    /// nil when `url` is a Gaussian splat this app can open, otherwise the reason it is not.
    static func problem(with url: URL) -> String? {
        let name = url.lastPathComponent
        let ext = url.pathExtension.lowercased()
        guard extensions.contains(ext) else {
            return "\(name) is not a .ply, .splat or .spz file"
        }
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            return "\(name) can't be read"
        }
        switch ext {
        case "ply":
            guard let header = PLYHeader(url: url) else { return "\(name) is not a valid PLY file" }
            // A point cloud PLY (x y z red green blue) has none of a splat's properties.
            let needed = ["opacity", "scale_0", "rot_0"]
            guard needed.allSatisfy(header.properties.contains) else {
                return "\(name) is a point cloud, not a Gaussian splat"
            }
        case "splat":
            let size = byteCount(url) ?? 0
            guard size > 0, size % 32 == 0 else { return "\(name) is not a valid .splat file" }
        default:
            break
        }
        return nil
    }

    /// How many splats the file holds, when its header or size says so.
    static func expectedCount(_ url: URL) -> Int? {
        switch url.pathExtension.lowercased() {
        case "ply": return PLYHeader(url: url)?.vertexCount
        case "splat": return byteCount(url).map { Int($0 / 32) }
        default: return nil
        }
    }

    static func byteCount(_ url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
    }

    /// The files to add for a dropped or chosen item: a file itself, or the splats in a
    /// folder and in its subfolders one level down. Of a .ply and a .splat with the same
    /// name, the .ply is kept: it carries view-dependent colour.
    static func expand(_ url: URL) -> [URL] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else { return [url] }
        guard isDirectory.boolValue else { return [url] }
        var found: [URL] = []
        let fm = FileManager.default
        let children = (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: [.isDirectoryKey],
                                                     options: [.skipsHiddenFiles])) ?? []
        for child in children.sorted(by: { $0.path < $1.path }) {
            let childIsDirectory = (try? child.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            if childIsDirectory {
                let grandchildren = (try? fm.contentsOfDirectory(at: child, includingPropertiesForKeys: nil,
                                                                  options: [.skipsHiddenFiles])) ?? []
                found += grandchildren.filter { problem(with: $0) == nil }
            } else if problem(with: child) == nil {
                found.append(child)
            }
        }
        var byStem: [String: URL] = [:]
        for file in found {
            let stem = file.deletingPathExtension().path
            if let existing = byStem[stem], existing.pathExtension.lowercased() == "ply" { continue }
            byStem[stem] = file
        }
        return byStem.values.sorted { $0.path < $1.path }
    }
}

struct PLYHeader {
    let vertexCount: Int?
    let properties: Set<String>

    init?(url: URL) {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: 64 * 1024),
              let text = String(data: data.prefix(64 * 1024), encoding: .isoLatin1),
              text.hasPrefix("ply"),
              let end = text.range(of: "end_header") else { return nil }
        var count: Int?
        var properties = Set<String>()
        for line in text[..<end.lowerBound].split(whereSeparator: \.isNewline) {
            let tokens = line.split(separator: " ")
            if tokens.first == "element", tokens.count >= 3, tokens[1] == "vertex" {
                count = Int(tokens[2])
            } else if tokens.first == "property", let last = tokens.last {
                properties.insert(String(last))
            }
        }
        vertexCount = count
        self.properties = properties
    }
}
