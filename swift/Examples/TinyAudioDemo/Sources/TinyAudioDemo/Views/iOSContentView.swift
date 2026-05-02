#if os(iOS)
  import SwiftUI

  struct iOSContentView: View {
    @ObservedObject var vm: TranscriberViewModel

    var body: some View {
      NavigationStack {
        TranscriptList(
          transcripts: vm.finalizedTranscripts,
          emptyPrompt: "Tap Record to start listening."
        )
        .navigationTitle("Tiny Audio")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
          ToolbarItem(placement: .topBarTrailing) {
            Button("Clear") { vm.clearTranscripts() }
              .buttonStyle(.borderedProminent)
              .disabled(vm.finalizedTranscripts.isEmpty)
          }
        }
        .safeAreaInset(edge: .bottom) {
          bottomBar
        }
        .alert(
          "Couldn't load model",
          isPresented: Binding(
            get: { vm.blockingError != nil },
            set: { if !$0 { vm.blockingError = nil } }
          ),
          actions: {
            Button("OK") { vm.blockingError = nil }
          },
          message: {
            Text(vm.blockingError ?? "")
          }
        )
      }
    }

    @ViewBuilder
    private var bottomBar: some View {
      VStack(spacing: 8) {
        if vm.permissionDenied {
          PermissionBanner()
            .padding(.horizontal, 16)
        }
        if let transient = vm.transientError {
          Text(transient)
            .font(.caption)
            .foregroundStyle(.orange)
            .padding(.horizontal, 16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        RecordButton(
          isListening: vm.isListening,
          isEnabled: vm.loadState == .ready,
          action: {
            Task {
              if vm.isListening {
                await vm.stopMic()
              } else {
                await vm.startMic()
              }
            }
          }
        )
      }
      .padding(.vertical, 12)
      .background(.bar)
    }
  }
#endif
