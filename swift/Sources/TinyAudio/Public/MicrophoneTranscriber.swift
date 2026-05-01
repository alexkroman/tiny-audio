// swift/Sources/TinyAudio/Public/MicrophoneTranscriber.swift
@preconcurrency import AVFoundation
import Foundation

/// A live-mic ASR actor that wires `AVAudioEngine` + Silero VAD + `Transcriber`.
///
/// Audio from the default input device is resampled to 16 kHz mono Float32 via
/// `AVAudioConverter`, sliced into `SileroVAD.frameSize`-sample frames, and fed
/// through `VADStreamer`. When the streamer emits a `.offset(audio:)` event the
/// accumulated utterance is dispatched to `Transcriber.transcribeStream`, and
/// each text delta is surfaced through `events` as a `.partial` followed by a
/// final `.final` event.
///
/// Per-utterance transcription failures emit an `.error(Error)` event rather
/// than terminating the stream, so a single bad utterance never ends the
/// session.
///
/// ## Lifecycle
/// ```swift
/// let transcriber = try await Transcriber.load()
/// let mic = try MicrophoneTranscriber(transcriber: transcriber)
/// try await mic.start()
/// for await event in mic.events {
///     switch event {
///     case .partial(let id, let delta): ...
///     case .final(let id, let text):   ...
///     case .error(let err):            ...
///     }
/// }
/// await mic.stop()
/// ```
public actor MicrophoneTranscriber {
    // MARK: - Public types

    /// Events emitted on `events`.
    public enum Event: Sendable {
        /// A partial text delta for the utterance identified by `utteranceID`.
        case partial(utteranceID: UUID, delta: String)
        /// The complete transcript for the utterance identified by `utteranceID`.
        case final(utteranceID: UUID, text: String)
        /// A per-utterance error. The session continues.
        case error(Error)
    }

    // MARK: - Public properties

    /// Non-throwing async stream of `Event` values.
    ///
    /// `nonisolated` so callers can subscribe without an `await`.
    public nonisolated let events: AsyncStream<Event>

    // MARK: - Private state

    private let transcriber: Transcriber
    private let config: VADConfig
    private let vad: SileroVAD
    private let streamer: VADStreamer
    private let engine: AVAudioEngine = AVAudioEngine()

    private let continuation: AsyncStream<Event>.Continuation

    private var running: Bool = false
    /// Leftover samples that didn't fill a complete VAD frame yet.
    private var pendingFrameTail: [Float] = []

    // MARK: - Init

    /// Create a `MicrophoneTranscriber`.
    ///
    /// - Parameters:
    ///   - transcriber: A loaded `Transcriber` instance.
    ///   - config: VAD endpointer tuning. Defaults to `.default`.
    /// - Throws: `TinyAudioError.vadModelMissing` or `.mlxModuleLoadFailed`
    ///   if the bundled Silero VAD model cannot be loaded. Does **not** prompt
    ///   for microphone permission; call `start()` for that.
    public init(transcriber: Transcriber, vad config: VADConfig = .default) throws {
        self.transcriber = transcriber
        self.config = config
        self.vad = try SileroVAD()
        self.streamer = VADStreamer(vad: self.vad, config: config)

        var localCont: AsyncStream<Event>.Continuation!
        self.events = AsyncStream { c in localCont = c }
        self.continuation = localCont
    }

    // MARK: - Public lifecycle

    /// Request microphone permission and start capturing audio.
    ///
    /// - Throws:
    ///   - `TinyAudioError.micPermissionDenied` when the user denies access.
    ///   - `TinyAudioError.audioSessionConfigurationFailed` when `AVAudioEngine`
    ///     cannot start (e.g. no input device, audio session conflict).
    public func start() async throws {
        guard !running else { return }

        // 1. Request microphone permission (iOS 17+ / macOS 14+).
        let granted = await Self.requestPermission()
        guard granted else { throw TinyAudioError.micPermissionDenied }

        // 2. Determine device format and build a 16 kHz mono output format.
        let inputNode = engine.inputNode
        let inFormat = inputNode.outputFormat(forBus: 0)
        guard let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        ) else {
            throw TinyAudioError.audioSessionConfigurationFailed(
                underlying: AnyError(TinyAudioError.audioFormatUnsupported(
                    reason: "could not create 16 kHz mono AVAudioFormat"
                ))
            )
        }

        guard let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            throw TinyAudioError.audioSessionConfigurationFailed(
                underlying: AnyError(TinyAudioError.audioFormatUnsupported(
                    reason: "no AVAudioConverter available for \(inFormat) -> 16 kHz mono"
                ))
            )
        }

        // 3. Install tap. The tap fires on a private AVAudioEngine thread;
        //    we hop to actor isolation via a Task for any state mutation.
        //    [weak self] avoids a retain cycle: if the actor is deinitialized
        //    while a tap callback is in-flight the guard safely exits.
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buffer, _ in
            guard let self else { return }
            let samples: [Float]
            do {
                samples = try Self.convertBuffer(buffer, with: converter, outFormat: outFormat)
            } catch {
                Task { await self.emitError(error) }
                return
            }
            Task { await self.feed(samples: samples) }
        }

        // 4. Start the engine.
        do {
            try engine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            throw TinyAudioError.audioSessionConfigurationFailed(underlying: AnyError(error))
        }
        running = true
    }

    /// Stop capturing audio and finish the `events` stream.
    public func stop() async {
        if running {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
            running = false
        }
        continuation.finish()
    }

    // MARK: - Private: audio feeding and VAD dispatch

    /// Append new samples, slice into `SileroVAD.frameSize`-sample frames,
    /// run each frame through the VAD streamer.
    private func feed(samples: [Float]) {
        pendingFrameTail.append(contentsOf: samples)
        let frameSize = SileroVAD.frameSize
        let completeFrames = pendingFrameTail.count / frameSize
        for i in 0 ..< completeFrames {
            let start = i * frameSize
            let frame = Array(pendingFrameTail[start ..< start + frameSize])
            let vadEvents: [VADStreamer.Event]
            do {
                vadEvents = try streamer.process(frame)
            } catch {
                emitError(error)
                continue
            }
            for ev in vadEvents {
                handleStreamerEvent(ev)
            }
        }
        if completeFrames > 0 {
            pendingFrameTail.removeFirst(completeFrames * frameSize)
        }
    }

    private func handleStreamerEvent(_ event: VADStreamer.Event) {
        switch event {
        case .onset:
            // No public event for onset; we only surface .partial / .final / .error.
            break
        case let .offset(audio):
            dispatchUtterance(audio: audio)
        }
    }

    /// Spawn a child Task to transcribe one utterance and yield its events.
    ///
    /// The child Task captures `continuation` by value (it's a reference type
    /// that is `Sendable`-safe to yield from any context) and `transcriber`
    /// (a `Sendable` actor reference). No actor re-entry needed inside the Task.
    private func dispatchUtterance(audio: [Float]) {
        let id = UUID()
        let cont = continuation
        let transcriber = self.transcriber
        Task {
            do {
                var collected = ""
                let stream = transcriber.transcribeStream(
                    .samples(audio, sampleRate: 16_000),
                    options: .default
                )
                for try await delta in stream {
                    collected += delta
                    cont.yield(.partial(utteranceID: id, delta: delta))
                }
                cont.yield(.final(utteranceID: id, text: collected))
            } catch {
                cont.yield(.error(error))
            }
        }
    }

    private func emitError(_ error: Error) {
        continuation.yield(.error(error))
    }

    // MARK: - Permission

    private static func requestPermission() async -> Bool {
        await withCheckedContinuation { cont in
            AVAudioApplication.requestRecordPermission { granted in
                cont.resume(returning: granted)
            }
        }
    }

    // MARK: - Audio conversion

    /// Convert one tap-delivered `AVAudioPCMBuffer` to 16 kHz mono `[Float]`.
    ///
    /// Uses the same single-shot delivery pattern as `AudioResampler.toMono16k`.
    private static func convertBuffer(
        _ inputBuffer: AVAudioPCMBuffer,
        with converter: AVAudioConverter,
        outFormat: AVAudioFormat
    ) throws -> [Float] {
        let inFrames = AVAudioFrameCount(inputBuffer.frameLength)
        guard inFrames > 0 else { return [] }
        let ratio = outFormat.sampleRate / inputBuffer.format.sampleRate
        let outCapacity = AVAudioFrameCount(Double(inFrames) * ratio + 1024)
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity) else {
            throw TinyAudioError.audioFormatUnsupported(reason: "output buffer allocation failed")
        }

        // Wrap the one-shot delivery flag in a class so the @Sendable closure
        // captures a reference (same pattern as AudioResampler.swift).
        final class InputState: @unchecked Sendable { var delivered = false }
        let state = InputState()
        var convError: NSError?
        let status = converter.convert(to: outBuffer, error: &convError) { _, outStatus in
            if state.delivered {
                outStatus.pointee = .endOfStream
                return nil
            }
            state.delivered = true
            outStatus.pointee = .haveData
            return inputBuffer
        }
        if status == .error || convError != nil {
            throw TinyAudioError.audioFormatUnsupported(
                reason: "conversion failed: \(convError?.localizedDescription ?? "unknown")"
            )
        }
        let frameCount = Int(outBuffer.frameLength)
        guard frameCount > 0, let ch = outBuffer.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: ch, count: frameCount))
    }
}
