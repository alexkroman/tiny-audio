import Foundation
import Testing

@testable import CookbookDemo

private actor MockClassifier: IntentClassifying {
  private let scripted: [String: Intent]
  init(_ scripted: [String: Intent]) { self.scripted = scripted }
  func classify(_ utterance: String) async -> Intent {
    scripted[utterance.lowercased()] ?? .none
  }
}

@Suite("CommandPipeline")
struct CommandPipelineTests {

  @Test func fastPathSkipsClassifier() async {
    let recipe = Recipe(title: "T", ingredients: [], steps: ["a", "b", "c"])
    let vm = await RecipeViewModel(recipe: recipe)
    let mock = MockClassifier([:])  // empty: would return .none if called
    let pipeline = CommandPipeline(viewModel: vm, classifier: mock)

    await pipeline.handle(transcribedText: "next")
    #expect(await vm.currentStepIndex == 1)
  }

  @Test func nonFastPathHitsClassifier() async {
    let recipe = Recipe(title: "T", ingredients: [], steps: ["a", "b", "c"])
    let vm = await RecipeViewModel(recipe: recipe)
    let mock = MockClassifier([
      "set a timer for five minutes": .setTimer(seconds: 300)
    ])
    let pipeline = CommandPipeline(viewModel: vm, classifier: mock)

    await pipeline.handle(transcribedText: "set a timer for five minutes")
    #expect(await vm.timer?.totalSeconds == 300)
  }

  @Test func updatesLastHeardText() async {
    let recipe = Recipe(title: "T", ingredients: [], steps: ["a"])
    let vm = await RecipeViewModel(recipe: recipe)
    let pipeline = CommandPipeline(viewModel: vm, classifier: MockClassifier([:]))

    await pipeline.handle(transcribedText: "the dog is barking")
    #expect(await vm.lastHeardText == "the dog is barking")
  }

  @Test func emptyUtteranceDoesNothing() async {
    let recipe = Recipe(title: "T", ingredients: [], steps: ["a"])
    let vm = await RecipeViewModel(recipe: recipe)
    let pipeline = CommandPipeline(viewModel: vm, classifier: MockClassifier([:]))

    await pipeline.handle(transcribedText: "   ")
    #expect(await vm.lastHeardText == "")
  }
}
