import Foundation
import Testing

@testable import TinyAudio

@Suite("ChatSession smoke")
struct ChatSessionTests {
  @Test func generatesNonEmptyStringFromBundledModel() async throws {
    guard let bundle = Bundle.module.url(forResource: "Model", withExtension: nil) else {
      print("[ChatSessionTests] skipping: bundled model not available")
      return
    }
    let session = try await ChatSession.makeForTests(modelDirectory: bundle)
    let out = try await session.chat(prompt: "Say hello.", maxNewTokens: 16)
    #expect(!out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
  }

  @Test func rejectsEmptyPrompt() async throws {
    guard let bundle = Bundle.module.url(forResource: "Model", withExtension: nil) else {
      print("[ChatSessionTests] skipping: bundled model not available")
      return
    }
    let session = try await ChatSession.makeForTests(modelDirectory: bundle)
    do {
      _ = try await session.chat(prompt: "   ", maxNewTokens: 8)
      Issue.record("expected promptEmpty")
    } catch TinyAudioError.promptEmpty {
      // expected
    }
  }

  @Test func rejectsZeroMaxNewTokens() async throws {
    guard let bundle = Bundle.module.url(forResource: "Model", withExtension: nil) else {
      print("[ChatSessionTests] skipping: bundled model not available")
      return
    }
    let session = try await ChatSession.makeForTests(modelDirectory: bundle)
    do {
      _ = try await session.chat(prompt: "hi", maxNewTokens: 0)
      Issue.record("expected invalidArgument")
    } catch TinyAudioError.invalidArgument {
      // expected
    }
  }
}
