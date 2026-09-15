import SwiftUI
import UniformTypeIdentifiers
import simd

extension UTType {
    static let oasisFurniture = UTType(exportedAs: "com.oasisspaces.furniture")
}

/// A piece dragged out of the furniture library. Its own type, so the window's file drop
/// does not take it for a file.
struct FurnitureDrag: Codable, Transferable {
    let kind: FurnitureKind

    static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .oasisFurniture)
    }
}

/// Everything drawn over the view while editing: the selection's outline and handles,
/// the library, the inspector and the toolbar.
struct EditorOverlay: View {
    let model: ViewerModel
    private var editor: EditorModel { model.editor }

    var body: some View {
        ZStack {
            GizmoOverlay(gizmo: editor.gizmo, isWall: { if case .wall = editor.selection { true } else { false } }())
                .allowsHitTesting(false)
            VStack(spacing: 0) {
                Spacer(minLength: 64)
                HStack(alignment: .top, spacing: 0) {
                    if editor.showLibrary {
                        FurnitureLibrary(model: model)
                            .transition(.move(edge: .leading).combined(with: .opacity))
                    }
                    Spacer(minLength: 0)
                    if editor.selection != nil {
                        Inspector(editor: editor)
                            .transition(.move(edge: .trailing).combined(with: .opacity))
                    }
                }
                Spacer(minLength: 12)
                EditToolbar(model: model)
            }
            .padding(14)
        }
        .animation(.easeOut(duration: 0.18), value: editor.showLibrary)
        .animation(.easeOut(duration: 0.18), value: editor.selection)
    }
}

// MARK: - Toolbar

struct EditToolbar: View {
    let model: ViewerModel
    private var editor: EditorModel { model.editor }

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 2) {
                ToolButton(title: "Add", systemImage: "plus.square.on.square", isOn: editor.showLibrary,
                           help: "Open the furniture library") {
                    editor.showLibrary.toggle()
                }
                divider
                ToolButton(title: "Rotate", systemImage: "rotate.right", help: "Turn the selection 90° (] turns 15°)",
                           enabled: canShape) {
                    editor.rotateSelection(by: .pi / 2)
                }
                ToolButton(title: "Smaller", systemImage: "minus.magnifyingglass", help: "Shrink the selection by 10%",
                           enabled: canShape) {
                    scaleSelection(by: 0.9)
                }
                ToolButton(title: "Larger", systemImage: "plus.magnifyingglass", help: "Grow the selection by 10%",
                           enabled: canShape) {
                    scaleSelection(by: 1.1)
                }
                ToolButton(title: "Remove", systemImage: "trash", help: "Take the selection out of the room (⌫)",
                           enabled: canShape) {
                    editor.removeSelection()
                }
                divider
                ToolButton(title: "Undo", systemImage: "arrow.uturn.backward", help: "Undo (⌘Z)", enabled: editor.canUndo) {
                    editor.undo()
                }
                ToolButton(title: "Redo", systemImage: "arrow.uturn.forward", help: "Redo (⌘⇧Z)", enabled: editor.canRedo) {
                    editor.redo()
                }
                if editor.removedCount > 0 {
                    ToolButton(title: "Restore \(editor.removedCount)", systemImage: "arrow.counterclockwise.circle",
                               help: "Put back everything removed") {
                        editor.restoreRemoved()
                    }
                }
                divider
                Button {
                    editor.isEditing = false
                } label: {
                    Text("Done")
                        .font(.system(size: 12, weight: .semibold))
                        .padding(.horizontal, 14)
                        .frame(height: 30)
                        .background(Color.oasis, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
                .help("Stop editing (⌘E)")
                .padding(.leading, 4)
            }
            .padding(5)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))

            Text(hint)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(.ultraThinMaterial, in: Capsule())
        }
    }

    private var divider: some View {
        Rectangle().fill(Color.white.opacity(0.12)).frame(width: 1, height: 26).padding(.horizontal, 4)
    }

    private var canShape: Bool {
        switch editor.selection {
        case .object, .added: true
        default: false
        }
    }

    private var hint: String {
        switch editor.selection {
        case .object, .added:
            "Drag to move · drag a corner to resize, the top dot for height, the front dot to turn · ⇧ keeps proportions"
        case .wall:
            "Pick a colour in the panel to repaint this wall"
        case nil:
            "Click furniture or a wall to select it · drag empty space to look around · arrows still walk"
        }
    }

    private func scaleSelection(by factor: Float) {
        guard let size = editor.selectionSize else { return }
        editor.setSize(size * factor)
    }
}

private struct ToolButton: View {
    let title: String
    let systemImage: String
    var isOn = false
    let help: String
    var enabled = true
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            VStack(spacing: 2) {
                Image(systemName: systemImage).font(.system(size: 13, weight: .medium))
                Text(title).font(.system(size: 9.5, weight: .medium))
            }
            .frame(minWidth: 50, minHeight: 38)
            .padding(.horizontal, 2)
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(isOn ? Color.oasis.opacity(0.35) : Color.white.opacity(hovering && enabled ? 0.1 : 0)))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(enabled ? .primary : .tertiary)
        .disabled(!enabled)
        .onHover { hovering = $0 }
        .help(help)
    }
}

// MARK: - Library

struct FurnitureLibrary: View {
    let model: ViewerModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Furniture").font(.system(size: 13, weight: .semibold))
                Spacer()
                Button {
                    model.editor.showLibrary = false
                } label: {
                    Image(systemName: "xmark").font(.system(size: 10, weight: .bold)).frame(width: 20, height: 20)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Close the library (esc)")
            }
            Text("Drag a piece into the room, or click it to place it in front of you.")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            ScrollView {
                LazyVGrid(columns: [GridItem(.flexible(), spacing: 8), GridItem(.flexible(), spacing: 8)], spacing: 8) {
                    ForEach(FurnitureKind.allCases) { kind in
                        FurnitureCard(kind: kind) {
                            model.controller?.addFurnitureInFront(kind)
                        }
                    }
                }
                .padding(.bottom, 4)
            }
            .scrollIndicators(.never)
        }
        .padding(12)
        .frame(width: 236)
        .frame(maxHeight: 520)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

private struct FurnitureCard: View {
    let kind: FurnitureKind
    let place: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: place) {
            VStack(spacing: 5) {
                Image(systemName: kind.systemImage)
                    .font(.system(size: 22))
                    .foregroundStyle(Color.oasis)
                    .frame(height: 30)
                Text(kind.label)
                    .font(.system(size: 11, weight: .medium))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Text(footprint)
                    .font(.system(size: 9.5))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 9)
            .background(RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white.opacity(hovering ? 0.12 : 0.06)))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .draggable(FurnitureDrag(kind: kind)) {
            Label(kind.label, systemImage: kind.systemImage)
                .padding(8)
                .background(.ultraThinMaterial, in: Capsule())
        }
        .help("\(kind.label), \(footprint)")
    }

    private var footprint: String {
        "\(Int((kind.size.x * 100).rounded())) × \(Int((kind.size.y * 100).rounded())) cm"
    }
}

// MARK: - Inspector

struct Inspector: View {
    let editor: EditorModel
    @State private var lastColourChange = Date.distantPast

    static let wallPaints: [(String, String)] = [
        ("Chalk", "#F2F0EB"), ("Warm white", "#EDE4D3"), ("Greige", "#CFC5B6"), ("Sage", "#B4C2A5"),
        ("Mist blue", "#BCD0DC"), ("Blush", "#E6C8BE"), ("Ochre", "#D3A85E"), ("Terracotta", "#C47559"),
        ("Olive", "#7D8360"), ("Navy", "#2F3D55"), ("Charcoal", "#4A4A4C"), ("Forest", "#35503F"),
    ]
    static let fabrics: [(String, String)] = [
        ("Linen", "#D9CFBF"), ("Stone", "#9C968C"), ("Oak", "#B08457"), ("Walnut", "#6E4F35"),
        ("Sage", "#8FA283"), ("Teal", "#3F6E73"), ("Rust", "#A65A3A"), ("Mustard", "#C9A13F"),
        ("Navy", "#2F3D55"), ("Blush", "#D9A9A0"), ("Charcoal", "#3E3E40"), ("White", "#EFEDE8"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(editor.selectionTitle).font(.system(size: 13, weight: .semibold)).lineLimit(1)
                Spacer()
                Button {
                    editor.selection = nil
                } label: {
                    Image(systemName: "xmark").font(.system(size: 10, weight: .bold)).frame(width: 20, height: 20)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Deselect (esc)")
            }
            switch editor.selection {
            case .object, .added:
                dimensions
                rotation
                if case .added = editor.selection {
                    colours(title: "Colour", palette: Self.fabrics, allWalls: false)
                }
                HStack {
                    Button("Reset", systemImage: "arrow.counterclockwise") { editor.resetSelection() }
                    Spacer()
                    Button("Remove", systemImage: "trash", role: .destructive) { editor.removeSelection() }
                }
                .controlSize(.small)
            case .wall:
                if let size = editor.selectionSize {
                    row("Size") {
                        Text("\(SceneController.length(size.x)) wide × \(SceneController.length(size.z)) high")
                            .font(.system(size: 11.5)).monospacedDigit()
                    }
                }
                colours(title: "Paint", palette: Self.wallPaints, allWalls: true)
                Button("Original colour", systemImage: "arrow.counterclockwise") { editor.resetSelection() }
                    .controlSize(.small)
                    .disabled(editor.selectionColour == nil)
            case nil:
                EmptyView()
            }
        }
        .padding(12)
        .frame(width: 252)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    // Width, depth and height in centimetres.
    private var dimensions: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionTitle("Dimensions")
            HStack(spacing: 6) {
                dimensionField("W", axis: 0)
                dimensionField("D", axis: 1)
                dimensionField("H", axis: 2)
            }
        }
    }

    private func dimensionField(_ label: String, axis: Int) -> some View {
        let binding = Binding<Int>(
            get: { Int(((editor.selectionSize?[axis] ?? 0) * 100).rounded()) },
            set: { centimetres in
                guard var size = editor.selectionSize else { return }
                size[axis] = Float(max(centimetres, 1)) / 100
                editor.setSize(size)
            })
        return VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.system(size: 9.5, weight: .semibold)).foregroundStyle(.secondary)
            HStack(spacing: 2) {
                TextField(label, value: binding, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11.5).monospacedDigit())
                    .multilineTextAlignment(.trailing)
                Text("cm").font(.system(size: 10)).foregroundStyle(.secondary)
            }
        }
    }

    private var rotation: some View {
        let degrees = Binding<Int>(
            get: {
                let value = Int((editor.selectionYaw * 180 / .pi).rounded())
                return (value % 360 + 360) % 360
            },
            set: { editor.setYaw(Float($0) * .pi / 180) })
        return VStack(alignment: .leading, spacing: 6) {
            sectionTitle("Rotation")
            HStack(spacing: 6) {
                Button { editor.rotateSelection(by: -.pi / 12) } label: { Image(systemName: "rotate.left") }
                    .help("Turn 15° left ([)")
                TextField("Rotation", value: degrees, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11.5).monospacedDigit())
                    .multilineTextAlignment(.trailing)
                    .frame(width: 56)
                Text("°").foregroundStyle(.secondary)
                Button { editor.rotateSelection(by: .pi / 12) } label: { Image(systemName: "rotate.right") }
                    .help("Turn 15° right (])")
                Spacer()
                Button("90°") { editor.rotateSelection(by: .pi / 2) }
            }
            .controlSize(.small)
        }
    }

    private func colours(title: String, palette: [(String, String)], allWalls: Bool) -> some View {
        let current = editor.selectionColour?.uppercased()
        let picker = Binding<Color>(
            get: { Color(hex: editor.selectionColour ?? "#FFFFFF") },
            set: { colour in
                // A picker sends a stream of changes: one undo step for each pause.
                if Date().timeIntervalSince(lastColourChange) > 1 { editor.beginChange() }
                lastColourChange = Date()
                editor.setColour(colour.hex, continuous: true)
            })
        return VStack(alignment: .leading, spacing: 7) {
            sectionTitle(title)
            LazyVGrid(columns: Array(repeating: GridItem(.fixed(28), spacing: 7), count: 6), alignment: .leading, spacing: 7) {
                ForEach(palette, id: \.1) { name, hex in
                    Button {
                        editor.setColour(hex)
                    } label: {
                        Circle()
                            .fill(Color(hex: hex))
                            .frame(width: 24, height: 24)
                            .overlay(Circle().strokeBorder(Color.white.opacity(0.25), lineWidth: 1))
                            .padding(2)
                            .overlay(Circle().strokeBorder(current == hex ? Color.oasis : .clear, lineWidth: 2))
                    }
                    .buttonStyle(.plain)
                    .help(name)
                }
            }
            HStack {
                ColorPicker("Custom", selection: picker, supportsOpacity: false)
                    .font(.system(size: 11.5))
                Spacer()
                if allWalls {
                    Button("Paint all walls") { editor.paintAllWalls(editor.selectionColour) }
                        .controlSize(.small)
                        .disabled(editor.selectionColour == nil)
                        .help("Give every detected wall this colour")
                }
            }
        }
    }

    private func sectionTitle(_ text: String) -> some View {
        Text(text.uppercased()).font(.system(size: 9.5, weight: .semibold)).foregroundStyle(.secondary)
    }

    private func row<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(title)
            content()
        }
    }
}

// MARK: - Selection outline

struct GizmoOverlay: View {
    let gizmo: Gizmo?
    let isWall: Bool

    var body: some View {
        Canvas { context, _ in
            guard let gizmo else { return }
            var lines = Path()
            for edge in gizmo.edges where edge.count == 2 {
                lines.move(to: edge[0])
                lines.addLine(to: edge[1])
            }
            context.stroke(lines, with: .color(.black.opacity(0.45)), lineWidth: 3.5)
            context.stroke(lines, with: .color(isWall ? .white : .oasis), lineWidth: 1.6)
            for (handle, point) in gizmo.handles {
                let radius: CGFloat = handle == .rotate ? 7 : 5.5
                let rect = CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)
                let shape = handle == .height ? Path(roundedRect: rect, cornerRadius: 2) : Path(ellipseIn: rect)
                context.fill(shape, with: .color(.white))
                context.stroke(shape, with: .color(.oasis), lineWidth: 2)
                if handle == .rotate {
                    context.draw(Image(systemName: "arrow.clockwise").resizable(),
                                 in: rect.insetBy(dx: 2.5, dy: 2.5))
                }
            }
            for (text, point) in gizmo.labels {
                let label = context.resolve(Text(text).font(.system(size: 11, weight: .semibold)).monospacedDigit()
                    .foregroundStyle(.white))
                let size = label.measure(in: CGSize(width: 300, height: 40))
                let box = CGRect(x: point.x - size.width / 2 - 6, y: point.y - size.height / 2 - 3,
                                 width: size.width + 12, height: size.height + 6)
                context.fill(Path(roundedRect: box, cornerRadius: box.height / 2), with: .color(.black.opacity(0.62)))
                context.draw(label, at: point)
            }
        }
    }
}

// MARK: - Colours

extension Color {
    init(hex: String) {
        let rgb = SIMD3<Float>(hex: hex) ?? SIMD3(1, 1, 1)
        self.init(red: Double(rgb.x), green: Double(rgb.y), blue: Double(rgb.z))
    }

    var hex: String {
        let resolved = NSColor(self).usingColorSpace(.sRGB) ?? .white
        return SIMD3(Float(resolved.redComponent), Float(resolved.greenComponent), Float(resolved.blueComponent)).hex
    }
}
