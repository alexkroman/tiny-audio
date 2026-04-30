import AVFoundation
import Foundation

/// Decode any `AudioInput` into 16 kHz mono `Float32` samples.
struct AudioDecoder {
    static func decode(_ input: AudioInput) throws -> [Float] {
        switch input {
        case let .file(url):
            return try decodeFile(url)
        case let .pcm(buffer):
            return try AudioResampler.toMono16k(buffer)
        case let .samples(samples, sampleRate):
            return try decodeSamples(samples, sampleRate: sampleRate)
        }
    }

    private static func decodeFile(_ url: URL) throws -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw TinyAudioError.audioFormatUnsupported(reason: "could not open \(url.lastPathComponent): \(error.localizedDescription)")
        }
        let frameCount = AVAudioFrameCount(file.length)
        guard frameCount > 0 else { throw TinyAudioError.audioEmpty }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frameCount) else {
            throw TinyAudioError.audioFormatUnsupported(reason: "could not allocate read buffer")
        }
        try file.read(into: buffer)
        return try AudioResampler.toMono16k(buffer)
    }

    private static func decodeSamples(_ samples: [Float], sampleRate: Double) throws -> [Float] {
        guard !samples.isEmpty else { throw TinyAudioError.audioEmpty }
        if sampleRate == AudioResampler.targetSampleRate {
            return samples
        }
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw TinyAudioError.audioFormatUnsupported(reason: "could not allocate input buffer")
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        let ch = buffer.floatChannelData![0]
        samples.withUnsafeBufferPointer { src in
            ch.update(from: src.baseAddress!, count: samples.count)
        }
        return try AudioResampler.toMono16k(buffer)
    }
}
