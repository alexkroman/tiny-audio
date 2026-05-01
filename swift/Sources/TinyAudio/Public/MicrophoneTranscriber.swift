// swift/Sources/TinyAudio/Public/MicrophoneTranscriber.swift
@preconcurrency import AVFoundation
import Foundation

/// Live-microphone speech recognition with Silero VAD endpointing.
///
/// `MicrophoneTranscriber` wires `AVAudioEngine`, a Silero Voice Activity
/// Detector, and a ``Transcriber`` into a single actor.  Audio captured from
/// the default input device is resampled to 16 kHz mono Float32, sliced into
/// VAD frames, and fed through the internal VAD endpointer.  When the VAD declares an
/// utterance has ended, the accumulated audio is dispatched to
/// ``Transcriber/transcribe(_:options:)`` and the resulting text is surfaced
/// through ``events`` as a single ``Event/final(utteranceID:text:)`` message.
///
/// Per-utterance transcription failures emit ``Event/error(_:)`` rather than
/// terminating the stream, so a single bad utterance never ends the session.
///
/// ## Lifecycle
///
/// ```swift
/// let transcriber = try await Transcriber.load()
/// let mic = try MicrophoneTranscriber(transcriber: transcriber)
/// try await mic.start()
/// for await event in mic.events {
///     switch event {
///     case .final(let id, let text): print("\n[\(id)] \(text)")
///     case .error(let err):          print("Error: \(err)")
///     }
/// }
/// await mic.stop()
/// ```
public actor MicrophoneTranscriber {
  // MARK: - Public types

  /// Events emitted on ``MicrophoneTranscriber/events``.
  ///
  /// Each detected utterance produces exactly one ``final(utteranceID:text:)``
  /// event with a stable `utteranceID` (`UUID`). On failure the session
  /// emits ``error(_:)`` and continues — subsequent utterances still produce
  /// `final` events normally.
  ///
  /// `Event` is `Sendable` so it can be forwarded across actor boundaries
  /// without copying.  However, the associated `Error` in ``error(_:)`` is
  /// not itself constrained to `Sendable`; handle it promptly rather than
  /// storing it across async suspension points.
  public enum Event: Sendable {
    /// The complete, settled transcript for one utterance.
    ///
    /// Emitted once the VAD declares the utterance ended and the
    /// `Transcriber` has finished decoding.
    case final(utteranceID: UUID, text: String)
    /// A transcription failure for one utterance.
    ///
    /// The session continues after this event; subsequent utterances
    /// produce new events normally.  Check ``TinyAudioError`` for
    /// recoverable cases (e.g. ``TinyAudioError/audioEmpty``).
    case error(Error)
  }

  // MARK: - Public properties

  /// A non-throwing async stream that delivers ``Event`` values as speech is detected.
  ///
  /// The stream is `nonisolated` so callers can begin iterating from any
  /// actor context without an initial `await` on `MicrophoneTranscriber`.
  /// The stream remains open until ``stop()`` is called, which calls
  /// `continuation.finish()` and causes the `for await` loop to exit
  /// naturally.
  ///
  /// ## Example
  ///
  /// ```swift
  /// let mic = try MicrophoneTranscriber(transcriber: transcriber)
  /// try await mic.start()
  ///
  /// Task {
  ///     for await event in mic.events {
  ///         if case let .final(_, text) = event {
  ///             print("Utterance: \(text)")
  ///         }
  ///     }
  ///     print("Stream ended.")
  /// }
  ///
  /// // Later…
  /// await mic.stop()
  /// ```
  public nonisolated let events: AsyncStream<Event>

  // MARK: - Private state

  private let transcriber: Transcriber
  private let config: VADConfig
  private let vad: SileroVAD
  private let streamer: VADStreamer
  private let engine: AVAudioEngine = AVAudioEngine()

  private let continuation: AsyncStream<Event>.Continuation

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
    self.events = AsyncStream(bufferingPolicy: .unbounded) { c in localCont = c }
    self.continuation = localCont
    // Attach onTermination after all stored properties are initialized so
    // [weak self] is valid. The continuation is already live; setting
    // onTermination at the end of init is safe — it fires only when the
    // stream is cancelled or finished, which cannot happen before init returns.
    localCont.onTermination = { @Sendable [weak self] _ in
      Task { await self?.stop() }
    }
  }

  // MARK: - Public lifecycle

  /// Request microphone permission and begin capturing audio.
  ///
  /// This method is idempotent — calling it when the engine is already
  /// running is a no-op.  On first call it prompts the user for microphone
  /// permission (iOS / macOS system dialog) and then starts `AVAudioEngine`.
  ///
  /// Audio is captured from the default system input device.  The engine
  /// resamples to 16 kHz mono Float32, feeds frames through the Silero VAD,
  /// and dispatches completed utterances to ``Transcriber/transcribe(_:options:)``.
  /// Results appear on ``events``.
  ///
  /// ## Example
  ///
  /// ```swift
  /// let mic = try MicrophoneTranscriber(transcriber: transcriber)
  /// try await mic.start()
  /// for await event in mic.events {
  ///     if case let .final(_, text) = event { print(text) }
  /// }
  /// ```
  ///
  /// - Throws: ``TinyAudioError/micPermissionDenied`` when the user denies
  ///   microphone access, or
  ///   ``TinyAudioError/audioSessionConfigurationFailed(underlying:)`` when
  ///   `AVAudioEngine` cannot start (e.g. no input device, audio session
  ///   conflict on iOS).
  public func start() async throws {
    guard !engine.isRunning else { return }

    // 1. Request microphone permission (iOS 17+ / macOS 14+).
    let granted = await Self.requestPermission()
    guard granted else { throw TinyAudioError.micPermissionDenied }

    // 2. Configure AVAudioSession for recording on iOS.
    //    Without this the inputNode delivers silence (or a session-conflict
    //    error). macOS has no AVAudioSession; the engine just works.
    #if os(iOS) || os(visionOS)
      do {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
        try session.setActive(true, options: [])
      } catch {
        throw TinyAudioError.audioSessionConfigurationFailed(underlying: AnyError(error))
      }
    #endif

    // 3. Determine device format and build a 16 kHz mono output format.
    let inputNode = engine.inputNode
    let inFormat = inputNode.outputFormat(forBus: 0)
    guard
      let outFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16_000,
        channels: 1,
        interleaved: false
      )
    else {
      throw TinyAudioError.audioSessionConfigurationFailed(
        underlying: AnyError(
          TinyAudioError.audioFormatUnsupported(
            reason: "could not create 16 kHz mono AVAudioFormat"
          ))
      )
    }

    guard let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
      throw TinyAudioError.audioSessionConfigurationFailed(
        underlying: AnyError(
          TinyAudioError.audioFormatUnsupported(
            reason: "no AVAudioConverter available for \(inFormat) -> 16 kHz mono"
          ))
      )
    }

    // 4. Install tap. The tap fires on a private AVAudioEngine thread;
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

    // 5. Start the engine.
    do {
      try engine.start()
    } catch {
      inputNode.removeTap(onBus: 0)
      throw TinyAudioError.audioSessionConfigurationFailed(underlying: AnyError(error))
    }
  }

  /// Stop capturing audio and finish the ``events`` stream.
  ///
  /// Removes the `AVAudioEngine` tap, stops the engine, and calls
  /// `continuation.finish()` so any active `for await` loop over ``events``
  /// exits cleanly.  Calling `stop()` when the engine is already stopped is
  /// safe and has no effect beyond finishing the stream a second time
  /// (which is itself a no-op in `AsyncStream`).
  ///
  /// Any utterances that are mid-transcription when `stop()` is called will
  /// still complete and emit their ``Event/final(utteranceID:text:)`` event
  /// before the stream loop can observe the finish, because the child `Task`
  /// for each utterance holds a reference to the continuation independently.
  public func stop() async {
    if engine.isRunning {
      engine.inputNode.removeTap(onBus: 0)
      engine.stop()
    }
    #if os(iOS) || os(visionOS)
      // Deactivate the audio session so other apps can resume their audio.
      // Failure here is non-fatal — the user is already done recording.
      try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    #endif
    continuation.finish()
  }

  // MARK: - Private: audio feeding and VAD dispatch

  /// Append new samples, slice into `SileroVAD.frameSize`-sample frames,
  /// run each frame through the VAD streamer.
  private func feed(samples: [Float]) {
    pendingFrameTail.append(contentsOf: samples)
    let frameSize = SileroVAD.frameSize
    let completeFrames = pendingFrameTail.count / frameSize
    for i in 0..<completeFrames {
      let start = i * frameSize
      let frame = Array(pendingFrameTail[start..<start + frameSize])
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
      // No public event for onset; we only surface .final / .error.
      break
    case .offset(let audio):
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
        let text = try await transcriber.transcribe(
          .samples(audio, sampleRate: 16_000),
          options: .default
        )
        cont.yield(.final(utteranceID: id, text: text))
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
  /// The converter is reused across tap callbacks. After delivering this
  /// call's input we signal `.noDataNow` (NOT `.endOfStream`) so the
  /// converter stays alive for the next call. Sending `.endOfStream` would
  /// finalize the converter and every subsequent `convert()` would return
  /// zero samples — see commit history for the bug this guards against.
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

    // Single-shot delivery: hand over this tap's buffer once, then signal
    // .noDataNow so the converter pauses without finalizing.
    final class InputState: @unchecked Sendable { var delivered = false }
    let state = InputState()
    var convError: NSError?
    let status = converter.convert(to: outBuffer, error: &convError) { _, outStatus in
      if state.delivered {
        outStatus.pointee = .noDataNow
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
