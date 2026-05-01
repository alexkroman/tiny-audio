import Foundation
import TinyAudio

/// `@unchecked Sendable` — this class is constructed once per `loadModel()`
/// call and only the `update(_:)` method writes to the AsyncStream
/// continuation, which itself is thread-safe. The closure passed to
/// `Transcriber.load(progress:)` runs on a non-main concurrency context;
/// `update(_:)` simply yields a Double through to the actor side.
private final class ProgressRelay: @unchecked Sendable {
    private let continuation: AsyncStream<Double>.Continuation

    let stream: AsyncStream<Double>

    init() {
        var cont: AsyncStream<Double>.Continuation!
        stream = AsyncStream { cont = $0 }
        continuation = cont
    }

    var callback: (Progress) -> Void {
        { [continuation] progress in
            continuation.yield(progress.fractionCompleted)
        }
    }

    func finish() {
        continuation.finish()
    }
}

@MainActor
final class TranscriberViewModel: ObservableObject {
    enum LoadState: Equatable {
        case idle
        case loading(progress: Double)
        case ready
        case error(String)
    }

    @Published var loadState: LoadState = .idle
    @Published var isListening: Bool = false
    @Published var liveTranscript: String = ""
    @Published var finalizedTranscripts: [String] = []
    @Published var lastError: String?

    private var transcriber: Transcriber?
    private var mic: MicrophoneTranscriber?
    private var consumeTask: Task<Void, Never>?

    func loadModel() async {
        loadState = .loading(progress: 0)
        let relay = ProgressRelay()

        // Drain the progress stream on the main actor while Transcriber.load runs.
        let progressTask = Task { @MainActor [weak self] in
            for await fraction in relay.stream {
                self?.loadState = .loading(progress: fraction)
            }
        }

        do {
            let t = try await Transcriber.load(progress: relay.callback)
            relay.finish()
            await progressTask.value
            transcriber = t
            loadState = .ready
        } catch {
            relay.finish()
            await progressTask.value
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
                    case let .partial(utteranceID: _, delta: delta):
                        self.liveTranscript += delta
                    case let .final(utteranceID: _, text: text):
                        self.finalizedTranscripts.append(text)
                        self.liveTranscript = ""
                    case let .error(err):
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
