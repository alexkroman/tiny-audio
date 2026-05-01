// swift/Tests/TinyAudioTests/ProfilingTests.swift
//
// Phase-level timing for `Transcriber.transcribe` on a representative audio
// sample. Gated behind `TINY_AUDIO_PROFILE=1` so it doesn't run on every CI
// build. Prints a breakdown table to stdout.
//
// Run with:
//   TINY_AUDIO_PROFILE=1 swift test --package-path swift --filter Profiling

import Foundation
import Testing
@_spi(Testing) @testable import TinyAudio

@Suite("Profiling")
struct ProfilingTests {
    @Test func phaseBreakdown() async throws {
        guard ProcessInfo.processInfo.environment["TINY_AUDIO_PROFILE"] == "1" else {
            print("Skipping ProfilingTests: set TINY_AUDIO_PROFILE=1 to run.")
            return
        }

        // Phase 1: cold load (one-time cost; not the hot path).
        let loadStart = Date()
        let transcriber = try await Transcriber.load(from: .defaultHub, progress: nil)
        let loadMs = Int(Date().timeIntervalSince(loadStart) * 1000)
        print("[load]    \(loadMs) ms (cold start: download check + verify + module load + warmup)")

        // Use the librispeech sample fixture (~6 s of audio).
        let url = try #require(
            Bundle.module.url(forResource: "librispeech_sample", withExtension: "wav", subdirectory: "Fixtures"),
            "librispeech_sample.wav fixture is missing"
        )

        // Run multiple iterations so JIT effects settle on the first run.
        // The first iteration includes any per-shape kernel compile; subsequent
        // iterations measure the steady-state hot path.
        for trial in 0 ..< 5 {
            let runStart = Date()
            let text = try await transcriber.transcribe(.file(url), options: .default)
            let runMs = Int(Date().timeIntervalSince(runStart) * 1000)
            let tag = trial == 0 ? "[trial 0 (warmup)]" : "[trial \(trial)]"
            print("\(tag) \(runMs) ms — \(text.prefix(80))…")
        }
    }
}
