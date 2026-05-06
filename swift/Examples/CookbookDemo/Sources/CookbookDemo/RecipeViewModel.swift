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
    case loading, cooking, micDenied
    case modelFailed(String)
  }
  enum ListeningState { case idle, hearing, thinking }

  let recipe: Recipe

  var phase: Phase = .loading
  var currentStepIndex: Int = 0
  var ingredientsVisible: Bool = false
  var timer: TimerState? = nil
  var groceryList: [String] = []
  var groceryOverlayVisible: Bool = false
  var recipeComplete: Bool = false
  var lastHeardText: String = ""
  var listeningState: ListeningState = .idle

  init(recipe: Recipe) { self.recipe = recipe }

  func apply(_ intent: Intent) {
    switch intent {
    case .nextStep:
      dismissOverlays()
      if currentStepIndex < recipe.steps.count - 1 {
        currentStepIndex += 1
      } else {
        recipeComplete = true
      }
    case .previousStep:
      dismissOverlays()
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
    case .none:
      break
    }
  }

  private func dismissOverlays() {
    ingredientsVisible = false
    groceryOverlayVisible = false
  }

  struct Snapshot: Equatable {
    let phase: Phase
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
      currentStepIndex: currentStepIndex,
      ingredientsVisible: ingredientsVisible,
      timer: timer,
      groceryList: groceryList,
      groceryOverlayVisible: groceryOverlayVisible,
      recipeComplete: recipeComplete
    )
  }
}
