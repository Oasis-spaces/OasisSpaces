import AppKit
import SwiftUI

@main
struct SplatViewerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("Splat Viewer", id: "main") {
            ContentView()
                .environment(Library.shared)
                .frame(minWidth: 760, minHeight: 520)
                // Files opened from Finder, the Dock icon or `open -a`. Delivered to the
                // window (which SwiftUI creates if needed); an app delegate taking the
                // open event instead leaves a cold launch with no window.
                .onOpenURL { Library.shared.add([$0]) }
        }
        .handlesExternalEvents(matching: ["*"])
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1280, height: 820)
        .commands { SplatCommands(library: Library.shared) }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        let arguments = CommandLine.arguments
        let environment = ProcessInfo.processInfo.environment
        // Paths go in the environment, not argv: AppKit opens file paths given as arguments.
        if let splat = environment["SPLATVIEWER_KEYTEST_SPLAT"], let log = environment["SPLATVIEWER_KEYTEST_LOG"] {
            SelfTest.runKeyTest(splat: URL(fileURLWithPath: splat), log: URL(fileURLWithPath: log))
        }
        if let log = environment["SPLATVIEWER_REPORT"] {
            SelfTest.reportWindows(to: URL(fileURLWithPath: log))
        }
        if let index = arguments.firstIndex(of: "--selftest"), arguments.count > index + 2 {
            SelfTest.run(outputDirectory: URL(fileURLWithPath: arguments[index + 1]),
                         splat: URL(fileURLWithPath: arguments[index + 2]))
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

struct SplatCommands: Commands {
    let library: Library

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("Open Splats…") { library.importing = true }
                .keyboardShortcut("o")
        }
        CommandGroup(before: .toolbar) {
            Button("Show Library") { library.showHome() }
                .keyboardShortcut("1")
            Button("Next Tab") { library.selectAdjacent(1) }
                .keyboardShortcut("]", modifiers: [.command, .shift])
            Button("Previous Tab") { library.selectAdjacent(-1) }
                .keyboardShortcut("[", modifiers: [.command, .shift])
            Divider()
        }
    }
}
