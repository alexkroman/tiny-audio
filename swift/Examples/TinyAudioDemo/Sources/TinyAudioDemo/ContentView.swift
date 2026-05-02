import SwiftUI

struct ContentView: View {
  @StateObject private var vm = TranscriberViewModel()
  @Environment(\.scenePhase) private var scenePhase

  var body: some View {
    platformView
      .task {
        // Run once on appear; intentionally NOT keyed on scenePhase so a
        // transient focus change during model load doesn't cancel the
        // long-running load.
        await vm.loadModel()
      }
      .onChange(of: scenePhase) { _, newPhase in
        if newPhase != .active {
          Task { await vm.stopMic() }
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
