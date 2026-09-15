import SwiftUI

/// The bar across the top of the window: the library, one tab per open splat, and +.
struct TabStrip: View {
    @Environment(Library.self) private var library

    var body: some View {
        HStack(spacing: 4) {
            Color.clear.frame(width: 70)   // the window's close, minimise and zoom buttons
            TabChip(title: "Library", systemImage: "square.grid.2x2",
                    isSelected: library.selected == nil, close: nil) {
                library.showHome()
            }
            .fixedSize(true)
            if !library.openTabs.isEmpty {
                Rectangle().fill(.separator).frame(width: 1, height: 16).padding(.horizontal, 4)
            }
            ForEach(library.openTabs, id: \.self) { id in
                if let item = library.item(id) {
                    TabChip(title: item.title, systemImage: "cube.transparent",
                            isSelected: library.selected == id,
                            close: { library.close(id) }) {
                        library.openTab(id)
                    }
                    .help(item.path)
                }
            }
            Button {
                library.importing = true
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 26, height: 26)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .help("Open splats (⌘O)")
            Spacer(minLength: 16)
        }
        .padding(.horizontal, 8)
        .frame(height: 40)
        .background {
            Rectangle()
                .fill(Color(nsColor: .windowBackgroundColor))
                .gesture(WindowDragGesture())
        }
        .overlay(alignment: .bottom) {
            Rectangle().fill(.separator).frame(height: 1)
        }
    }
}

struct TabChip: View {
    let title: String
    let systemImage: String
    let isSelected: Bool
    let close: (() -> Void)?
    let action: () -> Void
    private var fixed = false
    @State private var hovering = false
    @State private var hoveringClose = false

    init(title: String, systemImage: String, isSelected: Bool, close: (() -> Void)?, action: @escaping () -> Void) {
        self.title = title
        self.systemImage = systemImage
        self.isSelected = isSelected
        self.close = close
        self.action = action
    }

    func fixedSize(_ fixed: Bool) -> TabChip {
        var copy = self
        copy.fixed = fixed
        return copy
    }

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(isSelected ? Color.oasis : Color.secondary)
            Text(title)
                .font(.system(size: 12, weight: isSelected ? .semibold : .regular))
                .lineLimit(1)
                .truncationMode(.middle)
            if let close {
                Spacer(minLength: 0)
                Button(action: close) {
                    Image(systemName: "xmark")
                        .font(.system(size: 8.5, weight: .bold))
                        .frame(width: 16, height: 16)
                        .background(Circle().fill(Color.primary.opacity(hoveringClose ? 0.14 : 0)))
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .opacity(isSelected || hovering ? 1 : 0.45)
                .onHover { hoveringClose = $0 }
                .help("Close (⌘W)")
            }
        }
        .padding(.leading, 10)
        .padding(.trailing, close == nil ? 10 : 5)
        .frame(height: 28)
        .frame(minWidth: fixed ? nil : 96, maxWidth: fixed ? nil : 210)
        .background(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(Color.primary.opacity(isSelected ? 0.09 : (hovering ? 0.05 : 0)))
        )
        .contentShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        .onTapGesture(perform: action)
        .onHover { hovering = $0 }
        .animation(.easeOut(duration: 0.12), value: hovering)
    }
}
