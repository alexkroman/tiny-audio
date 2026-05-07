import Foundation
import Observation

struct TimerState: Equatable, Sendable {
  let totalSeconds: Int
  var secondsRemaining: Int
  /// Wall-clock time the timer should hit zero. Lets the tick task recover
  /// the correct remaining-seconds value across app sleeps without relying
  /// on dispatch precision.
  let endsAt: Date
}

@MainActor
@Observable
final class RecipeViewModel {
  enum Phase: Equatable {
    case loading, selecting, overview, cooking, micDenied
    case modelFailed(String)
  }
  enum ListeningState { case idle, hearing, thinking }

  let recipes: [Recipe]
  var recipe: Recipe?

  var phase: Phase = .loading
  var currentStepIndex: Int = 0
  var ingredientsVisible: Bool = false
  var timer: TimerState? = nil
  var groceryList: [String] = []
  var groceryOverlayVisible: Bool = false
  var recipeComplete: Bool = false
  var lastHeardText: String = ""
  var listeningState: ListeningState = .idle

  /// Catalog-driven init: starts at the recipe-selection screen with no recipe chosen.
  init(recipes: [Recipe]) {
    self.recipes = recipes
    self.recipe = nil
    self.phase = .selecting
  }

  /// Single-recipe convenience for tests and one-shot callers; jumps straight
  /// to the cooking phase with the recipe already selected.
  init(recipe: Recipe) {
    self.recipes = [recipe]
    self.recipe = recipe
    self.phase = .cooking
  }

  func setLastHeard(_ text: String) { lastHeardText = text }
  func setListeningState(_ state: ListeningState) { listeningState = state }

  func apply(_ intent: Intent) {
    switch intent {
    case .nextStep:
      dismissOverlays()
      if phase == .overview {
        phase = .cooking
        currentStepIndex = 0
        recipeComplete = false
        return
      }
      guard let recipe = recipe else { return }
      if currentStepIndex < recipe.steps.count - 1 {
        currentStepIndex += 1
      } else {
        recipeComplete = true
      }
    case .previousStep:
      dismissOverlays()
      if phase == .overview {
        recipe = nil
        phase = .selecting
        currentStepIndex = 0
        recipeComplete = false
        return
      }
      if currentStepIndex > 0 { currentStepIndex -= 1 }
    case .repeatStep:
      dismissOverlays()
    case .restart:
      dismissOverlays()
      currentStepIndex = 0
      recipeComplete = false
    case .readIngredients:
      groceryOverlayVisible = false
      ingredientsVisible = true
    case .setTimer(let seconds):
      timer = TimerState(
        totalSeconds: seconds,
        secondsRemaining: seconds,
        endsAt: Date().addingTimeInterval(TimeInterval(seconds))
      )
    case .cancelTimer:
      timer = nil
    case .addToGroceryList(let item):
      groceryList.append(item)
    case .showGroceryList:
      ingredientsVisible = false
      groceryOverlayVisible = true
    case .selectRecipe(let name):
      guard phase == .selecting else { return }
      guard let matched = matchRecipe(name: name) else { return }
      dismissOverlays()
      recipe = matched
      currentStepIndex = 0
      recipeComplete = false
      phase = .overview
    case .none:
      break
    }
  }

  private func dismissOverlays() {
    ingredientsVisible = false
    groceryOverlayVisible = false
  }

  private func matchRecipe(name: String) -> Recipe? {
    let slotTokens = name
      .lowercased()
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .split(whereSeparator: { $0.isWhitespace })
      .map(String.init)
    guard !slotTokens.isEmpty else { return nil }
    let candidates = recipes.filter { recipe in
      let titleLower = recipe.title.lowercased()
      return slotTokens.allSatisfy { titleLower.contains($0) }
    }
    return candidates.min(by: { $0.title.count < $1.title.count })
  }

  struct Snapshot: Equatable {
    let phase: Phase
    let recipeTitle: String?
    let currentStepIndex: Int
    let ingredientsVisible: Bool
    let timer: TimerState?
    let groceryList: [String]
    let groceryOverlayVisible: Bool
    let recipeComplete: Bool
  }

  func snapshot() -> Snapshot {
    Snapshot(
      phase: phase,
      recipeTitle: recipe?.title,
      currentStepIndex: currentStepIndex,
      ingredientsVisible: ingredientsVisible,
      timer: timer,
      groceryList: groceryList,
      groceryOverlayVisible: groceryOverlayVisible,
      recipeComplete: recipeComplete
    )
  }
}
