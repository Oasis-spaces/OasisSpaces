import SwiftUI
import ARKit
import SceneKit
import CaptureRules

/// Keeps one capture controller for as long as the screen is open.
final class ControllerBox: ObservableObject {
    let controller = CaptureController()
}

/// The camera with the live scan drawn over it, one guidance message at a
/// time, a top-down coverage map, and the record button.
struct CaptureScreen: View {
    @StateObject private var box = ControllerBox()

    var body: some View {
        CaptureContent(controller: box.controller, state: box.controller.state)
    }
}

private struct CaptureContent: View {
    let controller: CaptureController
    @ObservedObject var state: CaptureState
    @Environment(\.dismiss) private var dismiss
    @State private var showingMap = false

    var body: some View {
        ZStack {
            ARCameraView(controller: controller)
                .ignoresSafeArea()
            OutlineOverlay(state: state, session: controller.session, spec: controller.scene.spec)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                HStack(alignment: .top) {
                    Button { dismiss() } label: {
                        Image(systemName: "xmark")
                            .font(.headline)
                            .frame(width: 40, height: 40)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    .disabled(state.isRecording)
                    .opacity(state.isRecording ? 0 : 1)
                    Button { state.showOutlines.toggle() } label: {
                        Image(systemName: state.showOutlines ? "square.on.square.dashed" : "square.dashed")
                            .font(.headline)
                            .frame(width: 40, height: 40)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    Spacer()
                    Button { showingMap = true } label: {
                        CoverageMap(state: state, spec: controller.scene.spec)
                            .frame(width: 140, height: 140)
                    }
                    .buttonStyle(.plain)
                }
                GuidanceBanner(guidance: state.guidance, isRecording: state.isRecording, modelsReady: state.modelsReady)
                Spacer()
                if state.isRecording && !state.detected.isEmpty {
                    DetectedStrip(labels: state.detected)
                }
                ChecklistBar(state: state)
                RecordControls(state: state, start: controller.startRecording, stop: controller.stopRecording)
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)
            .padding(.bottom, 12)
        }
        .onAppear { controller.start() }
        .onDisappear { controller.pause() }
        .sheet(item: $state.result) { result in
            ReviewView(result: result, config: controller.config)
        }
        .sheet(isPresented: $showingMap) {
            RoomMapSheet(state: state, spec: controller.scene.spec)
        }
    }
}

struct ARCameraView: UIViewRepresentable {
    let controller: CaptureController

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.session = controller.session
        view.automaticallyUpdatesLighting = false
        view.scene.rootNode.addChildNode(controller.cloudNode)
        view.scene.rootNode.addChildNode(controller.mapNode)
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

struct GuidanceBanner: View {
    let guidance: Guidance?
    let isRecording: Bool
    var modelsReady = true

    var body: some View {
        let (text, icon, color) = content
        HStack(spacing: 10) {
            Image(systemName: icon).font(.title3)
            Text(text)
                .font(.headline)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(color.opacity(0.88), in: RoundedRectangle(cornerRadius: 16))
        .animation(.snappy, value: guidance?.rule)
    }

    private var content: (String, String, Color) {
        if !modelsReady {
            return ("Preparing the room detector… (the first time takes a minute)", "hourglass", .black)
        }
        guard let guidance else {
            return isRecording
                ? ("Looking good. Keep walking around the room", "checkmark.circle.fill", .green)
                : ("Point at the room, then press record", "camera.viewfinder", .black)
        }
        switch guidance.severity {
        case .stop: return (guidance.message, "exclamationmark.octagon.fill", .red)
        case .warn: return (guidance.message, "exclamationmark.triangle.fill", .orange)
        case .hint: return (guidance.message, "lightbulb.fill", .blue)
        }
    }
}

/// The two end-of-recording checks, ticked off live.
struct ChecklistBar: View {
    @ObservedObject var state: CaptureState

    var body: some View {
        if state.isRecording {
            HStack(spacing: 10) {
                Chip(done: state.coverage >= 0.75,
                     text: "Walls \(Int((state.coverage * 100).rounded()))%", icon: "square.dashed")
                Chip(done: state.floorSeen, text: "Floor", icon: "arrow.down.to.line")
                Chip(done: state.elapsed >= 20, text: timeText, icon: "timer")
            }
        }
    }

    private var timeText: String {
        let s = Int(state.elapsed)
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}

private struct Chip: View {
    let done: Bool
    let text: String
    let icon: String

    var body: some View {
        Label(text, systemImage: done ? "checkmark.circle.fill" : icon)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(done ? .green : .white)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())
    }
}

struct RecordControls: View {
    @ObservedObject var state: CaptureState
    let start: () -> Void
    let stop: () -> Void

    var body: some View {
        HStack {
            Text("\(state.map.objects.count) placed" + (state.depthOK ? "" : " · no depth"))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 100, alignment: .leading)
            Spacer()
            Button {
                state.isRecording ? stop() : start()
            } label: {
                ZStack {
                    Circle().stroke(.white, lineWidth: 5).frame(width: 78, height: 78)
                    RoundedRectangle(cornerRadius: state.isRecording ? 8 : 32)
                        .fill(.red)
                        .frame(width: state.isRecording ? 34 : 64, height: state.isRecording ? 34 : 64)
                }
            }
            .disabled(state.isFinishing)
            .animation(.snappy, value: state.isRecording)
            Spacer()
            Group {
                if state.isFinishing { ProgressView() }
            }
            .frame(width: 100, alignment: .trailing)
        }
    }
}

/// What has been recognised so far in this recording, most seen first.
struct DetectedStrip: View {
    let labels: [String]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(labels.prefix(14), id: \.self) { label in
                    Text(label)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(.ultraThinMaterial, in: Capsule())
                }
            }
        }
    }
}
