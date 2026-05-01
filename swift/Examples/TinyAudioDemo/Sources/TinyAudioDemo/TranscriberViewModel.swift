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
  @Published var permissionDenied: Bool = false

  /// One-shot blocking error surfaced via `.alert` (e.g. model load failure).
  @Published var blockingError: String?

  /// Transient per-utterance error surfaced as a caption that auto-dismisses.
  @Published var transientError: String?

  private var transcriber: Transcriber?
  private var mic: MicrophoneTranscriber?
  private var consumeTask: Task<Void, Never>?
  private var transientErrorClearTask: Task<Void, Never>?

  func loadModel() async {
    guard transcriber == nil else { return }
    loadState = .loading
    do {
      transcriber = try await Transcriber.load()
      loadState = .ready
    } catch {
      let message = String(describing: error)
      loadState = .error(message)
      blockingError = message
    }
  }

  func startMic() async {
    guard let transcriber else { return }
    do {
      let m = try MicrophoneTranscriber(transcriber: transcriber)
      mic = m
      try await m.start()
      isListening = true
      permissionDenied = false
      consumeTask = Task { @MainActor [weak self] in
        guard let self else { return }
        for await event in m.events {
          switch event {
          case .final(utteranceID: _, let text):
            self.finalizedTranscripts.append(text)
          case .error(let err):
            self.surfaceTransientError(String(describing: err))
          }
        }
      }
    } catch TinyAudioError.micPermissionDenied {
      permissionDenied = true
    } catch {
      blockingError = String(describing: error)
    }
  }

  func stopMic() async {
    await mic?.stop()
    consumeTask?.cancel()
    consumeTask = nil
    mic = nil
    isListening = false
  }

  func clearTranscripts() {
    finalizedTranscripts.removeAll()
  }

  private func surfaceTransientError(_ message: String) {
    transientError = message
    transientErrorClearTask?.cancel()
    transientErrorClearTask = Task { @MainActor [weak self] in
      try? await Task.sleep(for: .seconds(4))
      guard !Task.isCancelled else { return }
      self?.transientError = nil
    }
  }
}
