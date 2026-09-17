import SwiftUI
import CaptureRules

@main
struct OasisCaptureApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
                .preferredColorScheme(.dark)
        }
    }
}

/// Tips first, then the camera; the review sheet opens when a recording is saved.
struct RootView: View {
    @State private var capturing = false

    var body: some View {
        NavigationStack {
            TipsView(tips: RuleConfig.bundled().tips) { capturing = true }
                .navigationDestination(isPresented: $capturing) {
                    CaptureScreen()
                        .navigationBarBackButtonHidden(true)
                        .toolbar(.hidden, for: .navigationBar)
                }
        }
    }
}
