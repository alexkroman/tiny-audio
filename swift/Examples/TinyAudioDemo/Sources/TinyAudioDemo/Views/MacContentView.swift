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
      .frame(minWidth: 560, minHeight: 400)
      .toolbar {
        ToolbarItemGroup(placement: .navigation) {
          modelStatus
        }
        ToolbarItemGroup(placement: .primaryAction) {
          if vm.isListening {
            ListeningIndicator()
          }
          recordControl
          Button("Clear") { vm.clearTranscripts() }
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

    @ViewBuilder
    private var modelStatus: some View {
      switch vm.loadState {
      case .loading:
        HStack(spacing: 6) {
          ProgressView().controlSize(.small)
          Text("Loading…")
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }
      case .ready:
        Text("Ready")
          .font(.subheadline)
          .foregroundStyle(.secondary)
      case .error:
        Text("Load failed")
          .font(.subheadline)
          .foregroundStyle(.red)
      }
    }

    @ViewBuilder
    private var recordControl: some View {
      switch vm.loadState {
      case .ready:
        RecordButton(
          isListening: vm.isListening,
          isEnabled: true,
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
      default:
        RecordButton(isListening: false, isEnabled: false, action: {})
      }
    }
  }
#endif
