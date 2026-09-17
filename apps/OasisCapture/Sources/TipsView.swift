import SwiftUI
import CaptureRules

/// Before recording: what a good scan needs, and what makes it the best it can be.
struct TipsView: View {
    let tips: Tips
    let start: () -> Void
    @State private var showBest = true

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Scan a room")
                        .font(.largeTitle.bold())
                    Text("A minute or two of video is all it takes. A few things make the 3D model come out right.")
                        .foregroundStyle(.secondary)
                }

                TipSection(title: "For a good scan", subtitle: "Do these every time", tint: .green,
                           tips: tips.good)

                VStack(alignment: .leading, spacing: 12) {
                    Button {
                        withAnimation(.snappy) { showBest.toggle() }
                    } label: {
                        HStack {
                            Label("For the best scan", systemImage: "sparkles")
                                .font(.title3.bold())
                                .foregroundStyle(.yellow)
                            Spacer()
                            Image(systemName: showBest ? "chevron.up" : "chevron.down")
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                    Text("Worth the extra minute when you can")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    if showBest {
                        ForEach(tips.best) { TipRow(tip: $0, tint: .yellow) }
                    }
                }

                Text("While you record, the app watches how the phone moves and what it sees, and tells you straight away if something needs fixing.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .padding(24)
            .padding(.bottom, 96)
        }
        .safeAreaInset(edge: .bottom) {
            Button(action: start) {
                Label("I'm ready, open the camera", systemImage: "camera.viewfinder")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(.ultraThinMaterial)
        }
    }
}

private struct TipSection: View {
    let title: String
    let subtitle: String
    let tint: Color
    let tips: [Tip]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: "checkmark.seal")
                .font(.title3.bold())
                .foregroundStyle(tint)
            Text(subtitle)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            ForEach(tips) { TipRow(tip: $0, tint: tint) }
        }
    }
}

private struct TipRow: View {
    let tip: Tip
    let tint: Color

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: tip.icon)
                .font(.title3)
                .foregroundStyle(tint)
                .frame(width: 32, height: 32)
            VStack(alignment: .leading, spacing: 3) {
                Text(tip.title).font(.headline)
                Text(tip.detail)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 14))
    }
}
