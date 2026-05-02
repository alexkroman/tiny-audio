import SwiftUI
import TinyAudio
#if os(macOS)
  import AppKit
#endif

@main
struct TinyAudioDemoApp: App {
  init() {
    MLXBootstrap.ensureMetallibAvailable()
  }

  var body: some Scene {
    WindowGroup("Tiny Audio") {
      ContentView()
        .onAppear {
          #if os(macOS)
            // Bring the demo to the foreground on launch — without this,
            // the window can open behind the launching app (Xcode, terminal).
            NSApp.activate(ignoringOtherApps: true)
          #endif
        }
    }
  }
}
