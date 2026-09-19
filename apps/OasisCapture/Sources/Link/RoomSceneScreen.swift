import SwiftUI
import WebKit

/// A room as a mixed scene (a clean shell with the scanned furniture as
/// movable pieces), shown by the scene viewer the paired Mac serves on the
/// local network: the page, its libraries and the room's files all come from
/// the Mac, so it works with no internet.
struct RoomSceneScreen: View {
    let address: URL
    @Environment(\.dismiss) private var dismiss
    @State private var failed: String?

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.ignoresSafeArea()
            RoomWebView(address: address, failed: $failed)
                .ignoresSafeArea()
            if let failed {
                VStack(spacing: 10) {
                    Image(systemName: "wifi.exclamationmark").font(.largeTitle)
                    Text("Could not reach the Mac").font(.headline)
                    Text(failed).font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }
                .foregroundStyle(.white)
                .padding(30)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            Button { dismiss() } label: {
                Image(systemName: "xmark")
                    .font(.headline)
                    .frame(width: 40, height: 40)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .padding(.trailing, 14)
            .padding(.top, 6)
        }
        .preferredColorScheme(.dark)
    }
}

private struct RoomWebView: UIViewRepresentable {
    let address: URL
    @Binding var failed: String?

    func makeCoordinator() -> Coordinator { Coordinator(failed: $failed) }

    func makeUIView(context: Context) -> WKWebView {
        let view = WKWebView(frame: .zero, configuration: WKWebViewConfiguration())
        view.isOpaque = false
        view.backgroundColor = .black
        view.scrollView.isScrollEnabled = false          // one finger orbits the room, it does not scroll the page
        view.scrollView.bounces = false
        view.navigationDelegate = context.coordinator
        view.load(URLRequest(url: address, cachePolicy: .reloadIgnoringLocalCacheData))
        return view
    }

    func updateUIView(_ view: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKNavigationDelegate {
        @Binding var failed: String?
        init(failed: Binding<String?>) { _failed = failed }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            failed = "The room is shown by the Mac, over the same Wi-Fi. Open Splat Viewer on it and try again.\n(\(error.localizedDescription))"
        }
    }
}
