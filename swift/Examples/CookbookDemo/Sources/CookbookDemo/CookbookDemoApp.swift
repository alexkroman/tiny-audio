import SwiftUI

@main
struct CookbookDemoApp: App {
  var body: some Scene {
    WindowGroup("Cookbook") {
      ContentView()
        .frame(minWidth: 1100, minHeight: 700)
    }
    .windowResizability(.contentSize)
  }
}
