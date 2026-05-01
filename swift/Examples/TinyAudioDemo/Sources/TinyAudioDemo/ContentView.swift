import SwiftUI

struct ContentView: View {
  @StateObject private var vm = TranscriberViewModel()
  @Environment(\.scenePhase) private var scenePhase

  var body: some View {
    platformView
      .task(id: scenePhase) {
        // iOS revokes GPU access in the background; defer model load and pause
        // the mic when not active.
        if scenePhase == .active {
          await vm.loadModel()
        } else {
          await vm.stopMic()
        }
      }
  }

  @ViewBuilder
  private var platformView: some View {
    #if os(iOS)
      iOSContentView(vm: vm)
    #elseif os(macOS)
      MacContentView(vm: vm)
    #else
      Text("Unsupported platform")
    #endif
  }
}
