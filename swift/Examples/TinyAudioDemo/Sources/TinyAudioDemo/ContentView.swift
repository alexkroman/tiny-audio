import SwiftUI
import TinyAudio

struct ContentView: View {
    @StateObject private var vm = TranscriberViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("TinyAudio Demo")
                .font(.title)
                .bold()

            switch vm.loadState {
            case .idle:
                Button("Load Model") {
                    Task { await vm.loadModel() }
                }
                .buttonStyle(.borderedProminent)

            case let .loading(progress):
                VStack(alignment: .leading, spacing: 8) {
                    Text("Loading model — \(Int(progress * 100))%")
                        .font(.caption)
                    ProgressView(value: progress)
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
                        ProgressView()
                            .controlSize(.small)
                            .padding(.leading, 8)
                    }
                    Spacer()
                }

            case let .error(msg):
                Text("Load error: \(msg)")
                    .foregroundStyle(.red)
            }

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(Array(vm.finalizedTranscripts.enumerated()), id: \.offset) { _, text in
                        Text(text)
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

            if let err = vm.lastError {
                Text("Error: \(err)")
                    .foregroundStyle(.orange)
                    .font(.caption)
            }
        }
        .padding()
    }
}
