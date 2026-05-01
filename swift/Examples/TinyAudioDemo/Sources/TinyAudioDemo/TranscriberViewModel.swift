import Foundation
import TinyAudio

@MainActor
final class TranscriberViewModel: ObservableObject {
  enum LoadState: Equatable {
    case loading
    case ready
    case error(String)
  }

  @Published var loadState: LoadState = .loading
  @Published var isListening: Bool = false
  @Published var finalizedTranscripts: [String] = []
  @Published var lastError: String?

  private var transcriber: Transcriber?
  private var mic: MicrophoneTranscriber?
  private var consumeTask: Task<Void, Never>?

  /// Triggered automatically by ContentView on first appearance.
  func loadModel() async {
    guard transcriber == nil else { return }
    loadState = .loading
    do {
      transcriber = try await Transcriber.load(from: .defaultHub, progress: nil)
      loadState = .ready
    } catch {
      loadState = .error(String(describing: error))
    }
  }

  func startMic() async {
    guard let transcriber else { return }
    do {
      let m = try MicrophoneTranscriber(transcriber: transcriber)
      mic = m
      try await m.start()
      isListening = true
      consumeTask = Task { @MainActor [weak self] in
        guard let self else { return }
        for await event in m.events {
          switch event {
          case .final(utteranceID: _, let text):
            self.finalizedTranscripts.append(text)
          case .error(let err):
            self.lastError = String(describing: err)
          }
        }
      }
    } catch {
      lastError = String(describing: error)
    }
  }

  func stopMic() async {
    await mic?.stop()
    consumeTask?.cancel()
    consumeTask = nil
    mic = nil
    isListening = false
  }
}
