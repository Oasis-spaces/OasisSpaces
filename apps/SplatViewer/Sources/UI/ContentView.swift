import AppKit
import SwiftUI

struct ContentView: View {
    @Environment(Library.self) private var library
    @State private var dropTargeted = false
    @State private var shortcuts = WindowShortcuts()

    var body: some View {
        @Bindable var library = library
        VStack(spacing: 0) {
            TabStrip()
            ZStack {
                HomeView(dropTargeted: dropTargeted)
                    .opacity(library.selected == nil ? 1 : 0)
                    .allowsHitTesting(library.selected == nil)
                // Open splats stay loaded while another tab is shown; only the visible one draws.
                ForEach(library.openTabs, id: \.self) { id in
                    if let item = library.item(id) {
                        let isActive = library.selected == id
                        ViewerView(item: item, isActive: isActive, dropTargeted: dropTargeted && isActive)
                            .opacity(isActive ? 1 : 0)
                            .allowsHitTesting(isActive)
                    }
                }
            }
        }
        .ignoresSafeArea(.container, edges: .top)
        .tint(.oasis)
        .dropDestination(for: URL.self) { urls, _ in
            library.add(urls)
            return !urls.isEmpty
        } isTargeted: { dropTargeted = $0 }
        .fileImporter(isPresented: $library.importing,
                      allowedContentTypes: SplatFile.contentTypes,
                      allowsMultipleSelection: true) { result in
            if case .success(let urls) = result { library.add(urls) }
        }
        .onAppear {
            shortcuts.install(library: library)
            Thumbnailer.shared.refresh(library)
        }
        .onChange(of: library.items.count) { Thumbnailer.shared.refresh(library) }
    }
}

extension Color {
    /// Oasis green. Color.accentColor follows the system accent the user picked, so the
    /// app's own colour is named directly.
    static let oasis = Color("AccentColor")
}

/// Window shortcuts that depend on which tab is showing: ⌘W closes a splat tab (and only
/// on the library does it close the window), ⌘2…⌘9 pick a tab, ⌃⇥ cycles.
@MainActor
final class WindowShortcuts {
    private var monitor: Any?

    func install(library: Library) {
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            MainActor.assumeIsolated {
                guard !(event.window is NSPanel) else { return event }
                let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
                let key = event.charactersIgnoringModifiers?.lowercased()
                if flags == .command, key == "w", library.selected != nil {
                    library.closeSelected()
                    return nil
                }
                if flags == .command, let key, let digit = Int(key), (2...9).contains(digit) {
                    library.selectTab(at: digit - 2)
                    return nil
                }
                if flags.contains(.control), event.keyCode == 48 {
                    library.selectAdjacent(flags.contains(.shift) ? -1 : 1)
                    return nil
                }
                return event
            }
        }
    }
}
