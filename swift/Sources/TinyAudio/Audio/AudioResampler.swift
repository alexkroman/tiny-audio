import AVFoundation
import Foundation

/// Convert an `AVAudioPCMBuffer` to 16 kHz mono Float32 `[Float]`.
///
/// This is a thin wrapper around `AVAudioConverter`. It downmixes multi-channel
/// audio by averaging channels, then resamples to 16 kHz. Returns the raw
/// Float32 samples.
struct AudioResampler {
    static let targetSampleRate: Double = 16_000

    static func toMono16k(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        let inFormat = buffer.format
        let frameCount = AVAudioFrameCount(buffer.frameLength)
        guard frameCount > 0 else { return [] }

        let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        )!

        if inFormat == outFormat {
            return Self.copyMono(buffer)
        }

        guard let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            throw TinyAudioError.audioFormatUnsupported(reason: "no AVAudioConverter for \(inFormat) -> \(outFormat)")
        }

        let ratio = outFormat.sampleRate / inFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(frameCount) * ratio + 1024)
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity) else {
            throw TinyAudioError.audioFormatUnsupported(reason: "could not allocate output buffer")
        }

        var consumed = false
        var convError: NSError?
        let status = converter.convert(to: outBuffer, error: &convError) { _, outStatus in
            if consumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return buffer
        }

        if status == .error || convError != nil {
            throw TinyAudioError.audioFormatUnsupported(
                reason: "conversion failed: \(convError?.localizedDescription ?? "unknown")"
            )
        }
        return Self.copyMono(outBuffer)
    }

    private static func copyMono(_ buffer: AVAudioPCMBuffer) -> [Float] {
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0,
              let channels = buffer.floatChannelData
        else { return [] }
        let channelCount = Int(buffer.format.channelCount)
        if channelCount == 1 {
            return Array(UnsafeBufferPointer(start: channels[0], count: frameCount))
        }
        // Average across channels.
        var out = [Float](repeating: 0, count: frameCount)
        for c in 0 ..< channelCount {
            let ch = channels[c]
            for i in 0 ..< frameCount {
                out[i] += ch[i]
            }
        }
        let inv = 1.0 / Float(channelCount)
        for i in 0 ..< frameCount { out[i] *= inv }
        return out
    }
}
