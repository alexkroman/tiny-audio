import Foundation
import Testing

@testable import TinyAudio

@Suite("Transcriber.makeChatSession integration")
struct TranscriberChatSessionTests {
  @Test func loadAndChatRoundTrip() async throws {
    guard Bundle.module.url(forResource: "Model", withExtension: nil) != nil else {
      print("[TranscriberChatSessionTests] skipping: bundled model not available")
      return
    }
    let transcriber = try await Transcriber.load()
    let session = await transcriber.makeChatSession()
    let reply = try await session.chat(prompt: "Reply with just the word 'ok'.", maxNewTokens: 8)
    #expect(!reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
  }
}
