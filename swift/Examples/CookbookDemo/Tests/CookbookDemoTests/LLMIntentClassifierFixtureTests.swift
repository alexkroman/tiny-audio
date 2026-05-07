import Foundation
import Testing
import TinyAudio

@testable import CookbookDemo

@Suite("LLMIntentClassifier fixture")
struct LLMIntentClassifierFixtureTests {

  // 23 in-set + 5 noise = 28.
  private static let fixture: [(utterance: String, expected: Intent)] = [
    ("go to the next step", .nextStep),
    ("move on", .nextStep),
    ("can we keep going", .nextStep),
    ("go back one", .previousStep),
    ("can you go back", .previousStep),
    ("say that one more time", .repeatStep),
    ("read it again", .repeatStep),
    ("start the recipe over", .restart),
    ("what are the ingredients", .readIngredients),
    ("show me the ingredients", .readIngredients),
    ("set a timer for ten minutes", .setTimer(seconds: 600)),
    ("can you set a five minute timer", .setTimer(seconds: 300)),
    ("ninety second timer please", .setTimer(seconds: 90)),
    ("cancel that timer", .cancelTimer),
    ("stop the timer", .cancelTimer),
    ("add olive oil to the grocery list", .addToGroceryList(item: "olive oil")),
    ("put butter on my list", .addToGroceryList(item: "butter")),
    ("show my grocery list", .showGroceryList),
    ("show me what I need to buy", .showGroceryList),
    ("read me the shopping list", .showGroceryList),
    ("make some cookies", .selectRecipe(name: "cookies")),
    ("let's do pancakes today", .selectRecipe(name: "pancakes")),
    ("how about guacamole", .selectRecipe(name: "guacamole")),
    // Noise samples — all should classify as .none.
    ("the dog is barking again", .none),
    ("did you see the game last night", .none),
    ("hmm let me think", .none),
    ("oh that smells good", .none),
    ("uhhhh", .none),
  ]

  @Test func fixtureMatchRateAtLeastNinetyPercent() async throws {
    guard ProcessInfo.processInfo.environment["COOKBOOK_LLM_TEST"] == "1" else {
      print("[LLMIntentClassifierFixtureTests] skipping; set COOKBOOK_LLM_TEST=1 to run")
      return
    }

    let transcriber = try await Transcriber.load()
    let session = await transcriber.makeChatSession()
    let classifier = LLMIntentClassifier(session: session)

    var hits = 0
    for (utt, expected) in Self.fixture {
      let got = await classifier.classify(utt)
      let match = matches(got: got, expected: expected)
      if match {
        hits += 1
      } else {
        print("[fixture-miss] \"\(utt)\" expected \(expected) got \(got)")
      }
    }
    let rate = Double(hits) / Double(Self.fixture.count)
    print("[fixture] match rate = \(rate)")
    #expect(rate >= 0.90)
  }

  // For set_timer, accept ±20% on the seconds slot to absorb LLM rounding.
  private func matches(got: Intent, expected: Intent) -> Bool {
    switch (got, expected) {
    case (.setTimer(let g), .setTimer(let e)):
      let diff = Double(abs(g - e)) / Double(e)
      return diff <= 0.2
    case (.addToGroceryList(let g), .addToGroceryList(let e)):
      return g.lowercased().trimmingCharacters(in: .whitespaces)
        == e.lowercased().trimmingCharacters(in: .whitespaces)
    case (.selectRecipe(let g), .selectRecipe(let e)):
      return g.lowercased().trimmingCharacters(in: .whitespaces)
        == e.lowercased().trimmingCharacters(in: .whitespaces)
    default:
      return got == expected
    }
  }
}
