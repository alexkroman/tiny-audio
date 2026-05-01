// swift/Tests/TinyAudioTests/E2ETokenIDTests.swift
import Foundation
import Testing
@_spi(Testing) @testable import TinyAudio

@Suite("E2ETokenID")
struct E2ETokenIDTests {
    @Test func greedyTokenIDsMatchPython() async throws {
        guard ProcessInfo.processInfo.environment["TINY_AUDIO_E2E"] == "1" else {
            // Swift Testing doesn't have an equivalent of XCTSkip; print + return.
            print("Skipping E2ETokenIDTests: set TINY_AUDIO_E2E=1 to run (downloads ~460 MB).")
            return
        }

        let transcriber = try await Transcriber.load(from: .defaultHub, progress: nil)
        let url = Bundle.module.url(
            forResource: "librispeech_sample",
            withExtension: "wav",
            subdirectory: "Fixtures"
        )!

        let ids = try await transcriber.tokenIDsForTesting(.file(url), maxNewTokens: 20)

        let refURL = Bundle.module.url(
            forResource: "reference_token_ids",
            withExtension: "json",
            subdirectory: "Fixtures"
        )!
        let refData = try Data(contentsOf: refURL)
        let refRoot = try JSONSerialization.jsonObject(with: refData) as! [String: [Int]]
        let refIds = refRoot["token_ids"]!.map { Int32($0) }

        let n = min(20, refIds.count)
        let swiftPrefix = Array(ids.prefix(n))
        let refPrefix = Array(refIds.prefix(n))

        print("Swift IDs: \(swiftPrefix)")
        print("Ref   IDs: \(refPrefix)")

        #expect(swiftPrefix == refPrefix, "Swift token IDs differ from Python reference")
    }
}
