import Foundation
import Testing

@testable import CookbookDemo

@Suite("Intent")
struct IntentTests {
  @Test func decodesNextStep() {
    #expect(Intent.from(json: #"{"intent":"next_step"}"#) == .nextStep)
  }
  @Test func decodesPreviousStep() {
    #expect(Intent.from(json: #"{"intent":"previous_step"}"#) == .previousStep)
  }
  @Test func decodesRepeatStep() {
    #expect(Intent.from(json: #"{"intent":"repeat_step"}"#) == .repeatStep)
  }
  @Test func decodesRestart() {
    #expect(Intent.from(json: #"{"intent":"restart"}"#) == .restart)
  }
  @Test func decodesReadIngredients() {
    #expect(Intent.from(json: #"{"intent":"read_ingredients"}"#) == .readIngredients)
  }
  @Test func decodesSetTimer() {
    #expect(Intent.from(json: #"{"intent":"set_timer","seconds":300}"#) == .setTimer(seconds: 300))
  }
  @Test func setTimerWithMissingSecondsBecomesNone() {
    #expect(Intent.from(json: #"{"intent":"set_timer"}"#) == .none)
  }
  @Test func decodesCancelTimer() {
    #expect(Intent.from(json: #"{"intent":"cancel_timer"}"#) == .cancelTimer)
  }
  @Test func decodesAddToGroceryList() {
    #expect(
      Intent.from(json: #"{"intent":"add_to_grocery_list","item":"olive oil"}"#)
        == .addToGroceryList(item: "olive oil"))
  }
  @Test func addWithEmptyItemBecomesNone() {
    #expect(Intent.from(json: #"{"intent":"add_to_grocery_list","item":"  "}"#) == .none)
  }
  @Test func decodesShowGroceryList() {
    #expect(Intent.from(json: #"{"intent":"show_grocery_list"}"#) == .showGroceryList)
  }
  @Test func decodesNone() {
    #expect(Intent.from(json: #"{"intent":"none"}"#) == .none)
  }
  @Test func unknownIntentBecomesNone() {
    #expect(Intent.from(json: #"{"intent":"frobnicate"}"#) == .none)
  }
  @Test func malformedJSONBecomesNone() {
    #expect(Intent.from(json: "not json") == .none)
  }
  @Test func extractsFirstJSONObjectFromPreamble() {
    // Some LLM outputs wrap JSON in prose. The decoder should grab the first {...}.
    let raw = #"Sure! {"intent":"next_step"} — here you go."#
    #expect(Intent.from(json: raw) == .nextStep)
  }
}
