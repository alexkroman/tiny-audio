import Foundation
import TinyAudio

protocol IntentClassifying: Sendable {
  func classify(_ utterance: String) async -> Intent
}

/// Wraps `TinyAudio.ChatSession`. Builds a fixed-schema prompt, sends it to the
/// bundled Qwen3-0.6B, and parses the JSON reply via `Intent.from(json:)`.
/// Failures (LLM error, malformed output) collapse to `.none`.
final class LLMIntentClassifier: IntentClassifying, @unchecked Sendable {
  private let session: ChatSession

  init(session: ChatSession) { self.session = session }

  func classify(_ utterance: String) async -> Intent {
    let trimmed = utterance.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return .none }
    let prompt = Self.buildPrompt(utterance: trimmed)
    do {
      let reply = try await session.chat(prompt: prompt, maxNewTokens: 64)
      return Intent.from(json: reply)
    } catch {
      return .none
    }
  }

  static func buildPrompt(utterance: String) -> String {
    """
    You are a cooking-assistant command router. The user is cooking and just said:
    "\(utterance)"

    Classify into one intent. Respond with only a JSON object, no prose, no \
    explanation.
    Valid intents and slots:
    - {"intent":"next_step"}
    - {"intent":"previous_step"}
    - {"intent":"repeat_step"}
    - {"intent":"restart"}
    - {"intent":"read_ingredients"}
    - {"intent":"set_timer","seconds":<integer seconds>}
    - {"intent":"cancel_timer"}
    - {"intent":"add_to_grocery_list","item":"<short string>"}
    - {"intent":"show_grocery_list"}
    - {"intent":"select_recipe","name":"<recipe name as said>"}
    - {"intent":"none"}

    Examples:
    "go ahead" → {"intent":"next_step"}
    "set a timer for five minutes" → {"intent":"set_timer","seconds":300}
    "add olive oil to my list" → {"intent":"add_to_grocery_list","item":"olive oil"}
    "make some cookies" → {"intent":"select_recipe","name":"cookies"}
    "let's do pancakes" → {"intent":"select_recipe","name":"pancakes"}
    "pancakes" → {"intent":"select_recipe","name":"pancakes"}
    "guacamole" → {"intent":"select_recipe","name":"guacamole"}
    "the dog is barking" → {"intent":"none"}

    JSON:
    """
  }
}
