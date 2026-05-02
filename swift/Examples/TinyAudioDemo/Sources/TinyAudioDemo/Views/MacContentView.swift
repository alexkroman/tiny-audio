#if os(macOS)
  import SwiftUI

  struct MacContentView: View {
    @ObservedObject var vm: TranscriberViewModel

    var body: some View {
      VStack(spacing: 0) {
        if vm.permissionDenied {
          PermissionBanner()
            .padding(.horizontal, 16)
            .padding(.top, 12)
        }
        TranscriptList(
          transcripts: vm.finalizedTranscripts,
          emptyPrompt: "Click Record in the toolbar to start listening."
        )
        if let transient = vm.transientError {
          Text(transient)
            .font(.caption)
            .foregroundStyle(.orange)
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
      }
      .frame(minWidth: 360, minHeight: 400)
      .toolbar {
        ToolbarItemGroup(placement: .primaryAction) {
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
          Button("Clear") { vm.clearTranscripts() }
            .buttonStyle(.borderedProminent)
            .disabled(vm.finalizedTranscripts.isEmpty)
        }
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
#endif
