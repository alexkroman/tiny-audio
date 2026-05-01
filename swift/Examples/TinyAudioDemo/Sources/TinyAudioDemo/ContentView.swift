import SwiftUI
import TinyAudio

struct ContentView: View {
  @StateObject private var vm = TranscriberViewModel()
  @Environment(\.scenePhase) private var scenePhase

  var body: some View {
    NavigationStack {
      content
        .padding()
        .navigationTitle("TinyAudio Demo")
        #if os(macOS)
          .frame(minWidth: 480, minHeight: 320)
        #endif
        .task(id: scenePhase) {
          // iOS revokes GPU access in the background; defer model load
          // and pause the mic when not active.
          if scenePhase == .active {
            await vm.loadModel()
          } else {
            await vm.stopMic()
          }
        }
    }
  }

  @ViewBuilder
  private var content: some View {
    VStack(alignment: .leading, spacing: 16) {
      controlsSection
      Divider()
      transcriptScroll
      if let err = vm.lastError {
        Text("Error: \(err)").foregroundStyle(.orange).font(.caption)
      }
    }
  }

  @ViewBuilder
  private var controlsSection: some View {
    switch vm.loadState {
    case .loading:
      HStack(spacing: 12) {
        ProgressView().controlSize(.small)
        Text("Loading model…")
          .foregroundStyle(.secondary)
      }
    case .ready:
      HStack {
        if vm.isListening {
          Button("Stop") { Task { await vm.stopMic() } }
            .buttonStyle(.bordered)
            .tint(.red)
        } else {
          Button("Start Listening") { Task { await vm.startMic() } }
            .buttonStyle(.borderedProminent)
        }
        if vm.isListening {
          ProgressView().controlSize(.small).padding(.leading, 8)
        }
        Spacer()
      }
    case .error(let msg):
      Text("Load error: \(msg)").foregroundStyle(.red)
    }
  }

  @ViewBuilder
  private var transcriptScroll: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 12) {
        ForEach(vm.finalizedTranscripts.indices, id: \.self) { i in
          Text(vm.finalizedTranscripts[i])
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.gray.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        if !vm.liveTranscript.isEmpty {
          Text(vm.liveTranscript)
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.blue.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .foregroundStyle(.secondary)
        }
      }
    }
  }
}
