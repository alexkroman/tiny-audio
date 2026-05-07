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
    You convert a cook's spoken sentence into one JSON command.
    Output ONLY the JSON object. No other text.

    Schema (pick exactly one):
    {"intent":"next_step"}
    {"intent":"previous_step"}
    {"intent":"repeat_step"}
    {"intent":"restart"}
    {"intent":"read_ingredients"}
    {"intent":"set_timer","seconds":<int>}
    {"intent":"cancel_timer"}
    {"intent":"add_to_grocery_list","item":"<string>"}
    {"intent":"show_grocery_list"}
    {"intent":"select_recipe","name":"<string>"}
    {"intent":"none"}

    Examples:
    Input: go ahead
    Output: {"intent":"next_step"}
    Input: next
    Output: {"intent":"next_step"}
    Input: go back
    Output: {"intent":"previous_step"}
    Input: say that again
    Output: {"intent":"repeat_step"}
    Input: start over
    Output: {"intent":"restart"}
    Input: what do I need
    Output: {"intent":"read_ingredients"}
    Input: what are the ingredients
    Output: {"intent":"read_ingredients"}
    Input: set a timer for five minutes
    Output: {"intent":"set_timer","seconds":300}
    Input: timer for 30 seconds
    Output: {"intent":"set_timer","seconds":30}
    Input: two and a half minutes
    Output: {"intent":"set_timer","seconds":150}
    Input: one hour timer
    Output: {"intent":"set_timer","seconds":3600}
    Input: stop the timer
    Output: {"intent":"cancel_timer"}
    Input: cancel timer
    Output: {"intent":"cancel_timer"}
    Input: add olive oil to my list
    Output: {"intent":"add_to_grocery_list","item":"olive oil"}
    Input: put eggs on the grocery list
    Output: {"intent":"add_to_grocery_list","item":"eggs"}
    Input: show my grocery list
    Output: {"intent":"show_grocery_list"}
    Input: what's on my list
    Output: {"intent":"show_grocery_list"}
    Input: make some cookies
    Output: {"intent":"select_recipe","name":"cookies"}
    Input: let's do pancakes
    Output: {"intent":"select_recipe","name":"pancakes"}
    Input: pancakes
    Output: {"intent":"select_recipe","name":"pancakes"}
    Input: guacamole
    Output: {"intent":"select_recipe","name":"guacamole"}
    Input: the dog is barking
    Output: {"intent":"none"}
    Input: hmm
    Output: {"intent":"none"}

    Input: \(utterance)
    Output:
    """
  }
}
