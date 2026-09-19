import SwiftUI
import CaptureRules

@main
struct OasisCaptureApp: App {
    init() {
        AppLog.begin()
        // Start compiling the detector models now, so the camera is ready when opened.
        SceneRunner.shared.preload()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .preferredColorScheme(.dark)
        }
    }
}

/// Scan (tips, then the camera), Scans (send to the Mac, follow the analysis,
/// see results) and Mac (pairing).
struct RootView: View {
    /// `--autostart-scan` opens the camera at launch: `devicectl device process launch ... --autostart-scan`.
    @State private var capturing = ProcessInfo.processInfo.arguments.contains("--autostart-scan")
    @State private var tab = 0

    var body: some View {
        TabView(selection: $tab) {
            NavigationStack {
                TipsView(tips: RuleConfig.bundled().tips) { capturing = true }
                    .navigationDestination(isPresented: $capturing) {
                        CaptureScreen()
                            .navigationBarBackButtonHidden(true)
                            .toolbar(.hidden, for: .navigationBar)
                    }
            }
            .toolbar(capturing ? .hidden : .visible, for: .tabBar)
            .tabItem { Label("Scan", systemImage: "camera.viewfinder") }
            .tag(0)

            ScansView()
                .tabItem { Label("Scans", systemImage: "cube.transparent") }
                .tag(1)

            MacView()
                .tabItem { Label("Mac", systemImage: "laptopcomputer") }
                .tag(2)
        }
    }
}
