import SwiftUI

/// One open splat: the Metal view with its title, buttons, loading state and controls.
struct ViewerView: View {
    @Environment(Library.self) private var library
    let item: SplatItem
    let isActive: Bool
    let dropTargeted: Bool
    @State private var model = ViewerModel()

    var body: some View {
        ZStack {
            Color(red: 0.035, green: 0.04, blue: 0.04)
            SplatSceneView(item: item, isActive: isActive, model: model, library: library)

            switch model.phase {
            case .loading(let read, let total):
                LoadingCard(title: item.title, read: read, total: total)
            case .failed(let message):
                FailureCard(message: message) { library.close(item.id) }
            case .ready:
                EmptyView()
            }

            if model.editor.isEditing, model.phase == .ready {
                EditorOverlay(model: model)
            }

            VStack(spacing: 0) {
                topBar
                Spacer()
                HStack(alignment: .bottom) {
                    if model.showHelp && !model.editor.isEditing {
                        ControlsPanel(style: .overlay)
                            .transition(.opacity.combined(with: .move(edge: .bottom)))
                    }
                    Spacer()
                }
            }
            .padding(14)

            if dropTargeted {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(Color.oasis, style: StrokeStyle(lineWidth: 2, dash: [7, 5]))
                    .overlay(Text("Release to open in a new tab").font(.headline).padding(10)
                        .background(.ultraThinMaterial, in: Capsule()))
                    .padding(10)
                    .allowsHitTesting(false)
            }
        }
        .environment(\.colorScheme, .dark)
        .dropDestination(for: FurnitureDrag.self) { pieces, location in
            guard let piece = pieces.first, let controller = model.controller, model.editor.hasRoom else { return false }
            controller.dropFurniture(piece.kind, at: location)
            return true
        }
        .task(id: isActive) {
            // Show the controls when a splat first opens, then get out of the way.
            guard isActive, !model.helpAutoHidden else { return }
            try? await Task.sleep(for: .seconds(8))
            guard !Task.isCancelled else { return }
            model.helpAutoHidden = true
            withAnimation(.easeOut(duration: 0.3)) { model.showHelp = false }
        }
    }

    private var topBar: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(item.title)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                Text(subtitle)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            Spacer()
            if model.phase == .ready {
                editButton
            }
            HStack(spacing: 2) {
                OverlayButton(systemImage: "arrow.counterclockwise", help: "Back to the starting view (R)") {
                    model.controller?.resetView()
                }
                OverlayButton(systemImage: "keyboard", help: "Show or hide the controls (H)", isOn: model.showHelp) {
                    withAnimation(.easeOut(duration: 0.2)) { model.showHelp.toggle() }
                }
                OverlayButton(systemImage: "xmark", help: "Close this splat (⌘W)") {
                    library.close(item.id)
                }
            }
            .padding(3)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
    }

    private var editButton: some View {
        let editor = model.editor
        return Button {
            editor.isEditing.toggle()
            model.controller?.touch()
        } label: {
            Label(editor.isEditing ? "Editing" : "Edit room", systemImage: "square.and.pencil")
                .font(.system(size: 12, weight: .semibold))
                .padding(.horizontal, 11)
                .frame(height: 32)
                .background(editor.isEditing ? AnyShapeStyle(Color.oasis) : AnyShapeStyle(.ultraThinMaterial),
                            in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                .foregroundStyle(editor.isEditing ? .white : .primary)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!editor.hasRoom)
        .opacity(editor.hasRoom ? 1 : 0.5)
        .help(editor.hasRoom
              ? "Move, resize, remove and add furniture, and repaint walls (⌘E)"
              : "Editing needs the room's shapes.json and densify.json beside the splat")
    }

    private var subtitle: String {
        switch model.phase {
        case .ready:
            var parts = ["\(model.splatCount.formatted()) splats"]
            if model.startViewFrame != nil { parts.append("opened at the capture's best view") }
            return parts.joined(separator: " · ")
        case .loading: return "Loading…"
        case .failed: return "Couldn't open"
        }
    }
}

struct OverlayButton: View {
    let systemImage: String
    let help: String
    var isOn = false
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 13, weight: .medium))
                .frame(width: 30, height: 26)
                .background(RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(Color.white.opacity(isOn ? 0.16 : (hovering ? 0.1 : 0))))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.primary)
        .onHover { hovering = $0 }
        .help(help)
    }
}

struct LoadingCard: View {
    let title: String
    let read: Int
    let total: Int?

    var body: some View {
        VStack(spacing: 12) {
            if let total, total > 0 {
                ProgressView(value: min(Double(read) / Double(total), 1))
                    .progressViewStyle(.linear)
                    .frame(width: 240)
            } else {
                ProgressView().controlSize(.regular)
            }
            Text("Opening \(title)")
                .font(.system(size: 14, weight: .semibold))
            Text(detail)
                .font(.system(size: 11.5))
                .monospacedDigit()
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 22)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private var detail: String {
        guard read > 0 else { return "Reading the file…" }
        if let total, read >= total { return "Preparing \(total.formatted()) splats for the GPU…" }
        if let total { return "\(read.formatted()) of \(total.formatted()) splats" }
        return "\(read.formatted()) splats"
    }
}

struct FailureCard: View {
    let message: String
    let close: () -> Void

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 26))
                .foregroundStyle(.orange)
            Text("Couldn't open this splat")
                .font(.system(size: 14, weight: .semibold))
            Text(message)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 320)
            Button("Close", action: close)
                .padding(.top, 4)
        }
        .padding(24)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}
