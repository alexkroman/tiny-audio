import SwiftUI
import TinyAudio
#if os(macOS)
  import AppKit

  final class TinyAudioAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
      // Without this, the window can launch behind the launching app
      // (Xcode, Terminal). Calling from applicationDidFinishLaunching is
      // earlier than .onAppear and reliably wins the focus race.
      NSApp.setActivationPolicy(.regular)
      NSApp.activate(ignoringOtherApps: true)
      NSApp.windows.first?.makeKeyAndOrderFront(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(
      _ sender: NSApplication
    ) -> Bool {
      true
    }
  }
#endif

@main
struct TinyAudioDemoApp: App {
  #if os(macOS)
    @NSApplicationDelegateAdaptor(TinyAudioAppDelegate.self) private var appDelegate
  #endif

  init() {
    MLXBootstrap.ensureMetallibAvailable()
  }

  var body: some Scene {
    WindowGroup("Tiny Audio") {
      ContentView()
    }
    #if os(macOS)
      .defaultSize(width: 380, height: 520)
    #endif
  }
}
