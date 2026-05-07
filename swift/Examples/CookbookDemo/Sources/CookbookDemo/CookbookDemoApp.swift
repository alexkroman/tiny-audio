import AppKit
import SwiftUI

@main
struct CookbookDemoApp: App {
  init() {
    // Silence "Cannot index window tabs due to missing main bundle identifier"
    // — `swift run` executables don't carry a CFBundleIdentifier, which
    // AppKit's tab-state cache requires. We don't use tabs anyway.
    NSWindow.allowsAutomaticWindowTabbing = false
  }

  var body: some Scene {
    WindowGroup("Cookbook") {
      ContentView()
        .frame(minWidth: 1100, minHeight: 700)
    }
    .windowResizability(.contentSize)
  }
}
