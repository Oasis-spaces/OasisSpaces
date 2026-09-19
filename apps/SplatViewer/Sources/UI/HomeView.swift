import AppKit
import SwiftUI

/// The landing page: add splats by dropping or choosing them, and reopen the ones added before.
struct HomeView: View {
    let dropTargeted: Bool

    var body: some View {
        ScrollView {
            HomeContent(dropTargeted: dropTargeted)
        }
        .scrollIndicators(.automatic)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

/// The page without its scroll view (SelfTest renders this; ImageRenderer can't draw scroll views).
struct HomeContent: View {
    @Environment(Library.self) private var library
    let dropTargeted: Bool

    var body: some View {
            VStack(alignment: .leading, spacing: 30) {
                header
                DropZone(targeted: dropTargeted) { library.importing = true }
                PhoneCapturesSection()
                RoomsSection()
                if let notice = library.notice {
                    NoticeBanner(text: notice) { library.notice = nil }
                }
                librarySection
                ControlsPanel(style: .card)
            }
            .padding(.horizontal, 40)
            .padding(.top, 34)
            .padding(.bottom, 44)
            .frame(maxWidth: 1080)
            .frame(maxWidth: .infinity)
    }

    private var header: some View {
        HStack(spacing: 14) {
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .frame(width: 52, height: 52)
            VStack(alignment: .leading, spacing: 3) {
                Text("Splat Viewer")
                    .font(.system(size: 26, weight: .semibold))
                Text("Open Gaussian splats and walk through them with the keyboard.")
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var librarySection: some View {
        if library.items.isEmpty {
            HStack(spacing: 10) {
                Image(systemName: "sparkles.rectangle.stack")
                    .font(.system(size: 18))
                    .foregroundStyle(.tertiary)
                Text("Splats you add stay listed here, so you can reopen them with a click.")
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
        } else {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text("Your splats")
                        .font(.system(size: 17, weight: .semibold))
                    Text("\(library.items.count)")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 230, maximum: 330), spacing: 18)],
                          alignment: .leading, spacing: 18) {
                    ForEach(library.items) { item in
                        SplatCard(item: item)
                    }
                }
            }
        }
    }
}

struct DropZone: View {
    let targeted: Bool
    let choose: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(Color.oasis.opacity(targeted ? 0.2 : 0.11))
                    .frame(width: 66, height: 66)
                Image(systemName: targeted ? "arrow.down.circle.fill" : "arrow.down.doc")
                    .font(.system(size: 27, weight: .regular))
                    .foregroundStyle(Color.oasis)
                    .contentTransition(.symbolEffect(.replace))
            }
            VStack(spacing: 5) {
                Text(targeted ? "Release to open" : "Drop splat files or folders here")
                    .font(.system(size: 17, weight: .semibold))
                Text(".ply   .splat   .spz")
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 10) {
                Button(action: choose) {
                    Label("Choose Files…", systemImage: "folder")
                        .padding(.horizontal, 4)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                Text("⌘O")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 38)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(targeted ? Color.oasis.opacity(0.07) : Color.primary.opacity(0.025))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(targeted ? Color.oasis : Color.primary.opacity(0.16),
                              style: StrokeStyle(lineWidth: targeted ? 2 : 1.2, dash: [7, 5]))
        )
        .animation(.easeOut(duration: 0.15), value: targeted)
    }
}

struct NoticeBanner: View {
    let text: String
    let dismiss: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text(text)
                .font(.system(size: 12))
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
            Button(action: dismiss) {
                Image(systemName: "xmark").font(.system(size: 10, weight: .bold))
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.orange.opacity(0.1)))
    }
}

struct SplatCard: View {
    @Environment(Library.self) private var library
    let item: SplatItem
    @State private var hovering = false

    var body: some View {
        let exists = item.exists
        let isOpen = library.openTabs.contains(item.id)
        VStack(alignment: .leading, spacing: 0) {
            ZStack(alignment: .topLeading) {
                SplatThumbnail(url: library.thumbnailURL(for: item.id),
                               revision: library.thumbnailRevision[item.id, default: 0])
                if isOpen {
                    Text("OPEN")
                        .font(.system(size: 9, weight: .bold))
                        .tracking(0.6)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Capsule().fill(Color.oasis))
                        .foregroundStyle(.white)
                        .padding(9)
                }
            }
            .overlay(alignment: .topTrailing) {
                if hovering {
                    Button {
                        library.remove(item.id)
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 9, weight: .bold))
                            .frame(width: 22, height: 22)
                            .background(Circle().fill(.black.opacity(0.55)))
                            .foregroundStyle(.white)
                    }
                    .buttonStyle(.plain)
                    .padding(8)
                    .help("Remove from the library (the file stays on disk)")
                    .transition(.opacity)
                }
            }
            .clipShape(UnevenRoundedRectangle(topLeadingRadius: 12, topTrailingRadius: 12, style: .continuous))

            VStack(alignment: .leading, spacing: 3) {
                Text(item.title)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                Text(detail(exists: exists))
                    .font(.system(size: 11))
                    .foregroundStyle(exists ? Color.secondary : Color.orange)
                    .lineLimit(1)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
        }
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color.primary.opacity(hovering ? 0.2 : 0.09), lineWidth: 1)
        )
        .shadow(color: .black.opacity(hovering ? 0.14 : 0.05), radius: hovering ? 12 : 4, y: hovering ? 4 : 1)
        .opacity(exists ? 1 : 0.6)
        .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .onHover { hovering = $0 }
        .animation(.easeOut(duration: 0.14), value: hovering)
        .onTapGesture { if exists { library.openTab(item.id) } }
        .contextMenu {
            Button("Open") { library.openTab(item.id) }.disabled(!exists)
            Button("Show in Finder") { library.revealInFinder(item.id) }.disabled(!exists)
            Divider()
            Button("Remove from Library") { library.remove(item.id) }
        }
        .help(item.path)
    }

    private func detail(exists: Bool) -> String {
        guard exists else { return "File not found · \(item.fileName)" }
        var parts = [item.fileName]
        if let bytes = item.byteCount {
            parts.append(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file))
        }
        if let count = item.splatCount { parts.append("\(count.formatted()) splats") }
        return parts.joined(separator: " · ")
    }
}

struct SplatThumbnail: View {
    let url: URL
    let revision: Int

    var body: some View {
        Color.clear
            .aspectRatio(16 / 10, contentMode: .fit)
            .overlay {
                if let image = NSImage(contentsOf: url) {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    ZStack {
                        LinearGradient(colors: [Color(red: 0.11, green: 0.2, blue: 0.18),
                                                Color(red: 0.05, green: 0.08, blue: 0.08)],
                                       startPoint: .topLeading, endPoint: .bottomTrailing)
                        Image(systemName: "cube.transparent")
                            .font(.system(size: 30, weight: .light))
                            .foregroundStyle(.white.opacity(0.45))
                    }
                }
            }
            .clipped()
            .id(revision)
    }
}

/// The keyboard and mouse controls, as a card on the landing page or a panel over a splat.
struct ControlsPanel: View {
    enum Style { case card, overlay }
    let style: Style

    private var rows: [(keys: [String], action: String)] {
        var rows: [(keys: [String], action: String)] = [
            (["↑", "↓", "←", "→"], "Move forward, back, left and right"),
            (["⌘", "←", "→"], "Turn left and right"),
            (["⌘", "↑", "↓"], "Look up and down"),
            (["⌥", "↑", "↓"], "Move up and down"),
            (["⇧"], "Hold to move faster"),
            (["Drag"], "Look around"),
            (["Scroll"], "Walk forward and sideways"),
            (["R"], "Back to the starting view"),
            (["⌘", "E"], "Edit the room"),
        ]
        if style == .card {
            rows += [(["⌘", "O"], "Open splats"), (["⌘", "W"], "Close the splat"),
                     (["⌘", "1"], "Library"), (["⌃", "⇥"], "Next tab")]
        } else {
            rows += [(["H"], "Show or hide these controls"), (["⌘", "W"], "Close the splat")]
        }
        return rows
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Controls")
                .font(.system(size: style == .card ? 15 : 12, weight: .semibold))
            let columns = style == .card
                ? [GridItem(.flexible(), spacing: 24, alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                : [GridItem(.flexible(), alignment: .leading)]
            LazyVGrid(columns: columns, alignment: .leading, spacing: style == .card ? 10 : 7) {
                ForEach(rows.indices, id: \.self) { index in
                    HStack(spacing: 10) {
                        HStack(spacing: 3) {
                            ForEach(rows[index].keys, id: \.self) { KeyCap(label: $0, small: style == .overlay) }
                        }
                        .frame(width: style == .card ? 104 : 88, alignment: .leading)
                        Text(rows[index].action)
                            .font(.system(size: style == .card ? 12.5 : 11.5))
                            .foregroundStyle(style == .card ? Color.secondary : Color.primary.opacity(0.85))
                    }
                }
            }
        }
        .padding(style == .card ? 18 : 14)
        .background {
            if style == .card {
                RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color.primary.opacity(0.035))
            } else {
                RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.ultraThinMaterial)
            }
        }
        .frame(maxWidth: style == .card ? .infinity : 300, alignment: .leading)
    }
}

struct KeyCap: View {
    let label: String
    var small = false

    var body: some View {
        Text(label)
            .font(.system(size: small ? 10.5 : 11.5, weight: .medium, design: label.count > 1 ? .default : .rounded))
            .padding(.horizontal, label.count > 1 ? 6 : 0)
            .frame(minWidth: small ? 20 : 22, minHeight: small ? 19 : 21)
            .background(
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .fill(Color.primary.opacity(0.07))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.14), lineWidth: 0.8)
            )
    }
}
