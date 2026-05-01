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
        .navigationTitle("TinyAudio")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
          ToolbarItem(placement: .topBarTrailing) {
            Button("Clear") { vm.clearTranscripts() }
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
        controlRow
      }
      .padding(.vertical, 12)
      .background(.bar)
    }

    @ViewBuilder
    private var controlRow: some View {
      HStack(spacing: 12) {
        Spacer()
        if vm.isListening {
          ListeningIndicator()
        }
        switch vm.loadState {
        case .loading:
          HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text("Loading model…")
              .font(.subheadline)
              .foregroundStyle(.secondary)
          }
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
        case .error:
          RecordButton(isListening: false, isEnabled: false, action: {})
        }
        Spacer()
      }
      .padding(.horizontal, 16)
    }
  }
#endif
