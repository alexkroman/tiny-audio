import Foundation
import Testing

@testable import CookbookDemo

@MainActor
@Suite("RecipeViewModel.apply")
struct RecipeViewModelTests {
  private func makeVM() -> RecipeViewModel {
    let recipe = Recipe(
      title: "T",
      ingredients: ["a", "b"],
      steps: ["one", "two", "three"]
    )
    return RecipeViewModel(recipe: recipe)
  }

  @Test func nextStepAdvances() {
    let vm = makeVM()
    vm.apply(.nextStep)
    #expect(vm.currentStepIndex == 1)
  }

  @Test func nextStepAtLastStepShowsCompleteOverlay() {
    let vm = makeVM()
    vm.apply(.nextStep)
    vm.apply(.nextStep)  // index = 2 (last)
    vm.apply(.nextStep)
    #expect(vm.currentStepIndex == 2)
    #expect(vm.recipeComplete == true)
  }

  @Test func previousStepGoesBack() {
    let vm = makeVM()
    vm.apply(.nextStep)
    vm.apply(.previousStep)
    #expect(vm.currentStepIndex == 0)
  }

  @Test func previousAtFirstStepIsNoOp() {
    let vm = makeVM()
    vm.apply(.previousStep)
    #expect(vm.currentStepIndex == 0)
  }

  @Test func restartGoesToZero() {
    let vm = makeVM()
    vm.apply(.nextStep)
    vm.apply(.nextStep)
    vm.apply(.restart)
    #expect(vm.currentStepIndex == 0)
    #expect(vm.recipeComplete == false)
  }

  @Test func readIngredientsShowsPanel() {
    let vm = makeVM()
    vm.apply(.readIngredients)
    #expect(vm.ingredientsVisible == true)
  }

  @Test func navigationDismissesIngredientsPanel() {
    let vm = makeVM()
    vm.apply(.readIngredients)
    vm.apply(.nextStep)
    #expect(vm.ingredientsVisible == false)
  }

  @Test func setTimerStartsTimer() {
    let vm = makeVM()
    vm.apply(.setTimer(seconds: 300))
    #expect(vm.timer?.totalSeconds == 300)
  }

  @Test func setTimerReplacesExistingTimer() {
    let vm = makeVM()
    vm.apply(.setTimer(seconds: 60))
    vm.apply(.setTimer(seconds: 120))
    #expect(vm.timer?.totalSeconds == 120)
  }

  @Test func cancelTimerClears() {
    let vm = makeVM()
    vm.apply(.setTimer(seconds: 60))
    vm.apply(.cancelTimer)
    #expect(vm.timer == nil)
  }

  @Test func cancelTimerWithNoActiveTimerIsNoOp() {
    let vm = makeVM()
    vm.apply(.cancelTimer)
    #expect(vm.timer == nil)
  }

  @Test func addToGroceryListAppends() {
    let vm = makeVM()
    vm.apply(.addToGroceryList(item: "olive oil"))
    vm.apply(.addToGroceryList(item: "salt"))
    #expect(vm.groceryList == ["olive oil", "salt"])
  }

  @Test func showGroceryListSetsVisible() {
    let vm = makeVM()
    vm.apply(.showGroceryList)
    #expect(vm.groceryOverlayVisible == true)
  }

  @Test func navigationDismissesGroceryOverlay() {
    let vm = makeVM()
    vm.apply(.showGroceryList)
    vm.apply(.previousStep)
    #expect(vm.groceryOverlayVisible == false)
  }

  @Test func noneIsNoOp() {
    let vm = makeVM()
    let before = vm.snapshot()
    vm.apply(.none)
    #expect(vm.snapshot() == before)
  }

  private func makeCatalogVM() -> RecipeViewModel {
    let cookies = Recipe(
      title: "Chocolate Chip Cookies",
      ingredients: ["flour", "sugar"],
      steps: ["mix", "bake"]
    )
    let pancakes = Recipe(
      title: "Pancakes",
      ingredients: ["flour", "milk"],
      steps: ["mix", "cook"]
    )
    return RecipeViewModel(recipes: [cookies, pancakes])
  }

  @Test func catalogInitStartsInSelectingPhaseWithNoRecipe() {
    let vm = makeCatalogVM()
    #expect(vm.phase == .selecting)
    #expect(vm.recipe == nil)
  }

  @Test func selectRecipeMatchesByToken() {
    let vm = makeCatalogVM()
    vm.apply(.selectRecipe(name: "cookies"))
    #expect(vm.phase == .overview)
    #expect(vm.recipe?.title == "Chocolate Chip Cookies")
    #expect(vm.currentStepIndex == 0)
  }

  @Test func selectRecipeMatchesMultiTokenSlot() {
    let vm = makeCatalogVM()
    vm.apply(.selectRecipe(name: "chip cookies"))
    #expect(vm.recipe?.title == "Chocolate Chip Cookies")
  }

  @Test func selectRecipeIsCaseInsensitive() {
    let vm = makeCatalogVM()
    vm.apply(.selectRecipe(name: "PANCAKES"))
    #expect(vm.recipe?.title == "Pancakes")
  }

  @Test func selectRecipeNoMatchIsNoOp() {
    let vm = makeCatalogVM()
    vm.apply(.selectRecipe(name: "lasagna"))
    #expect(vm.phase == .selecting)
    #expect(vm.recipe == nil)
  }

  @Test func selectRecipeIgnoredOutsideSelectingPhase() {
    let vm = makeCatalogVM()
    vm.apply(.selectRecipe(name: "cookies"))  // → overview, recipe=cookies
    vm.apply(.selectRecipe(name: "pancakes"))  // ignored
    #expect(vm.recipe?.title == "Chocolate Chip Cookies")
    #expect(vm.phase == .overview)
  }

  @Test func nextStepFromOverviewEntersCooking() {
    let vm = makeCatalogVM()
    vm.apply(.selectRecipe(name: "cookies"))
    #expect(vm.phase == .overview)
    vm.apply(.nextStep)
    #expect(vm.phase == .cooking)
    #expect(vm.currentStepIndex == 0)
  }

  @Test func previousStepFromOverviewReturnsToSelection() {
    let vm = makeCatalogVM()
    vm.apply(.selectRecipe(name: "cookies"))
    vm.apply(.previousStep)
    #expect(vm.phase == .selecting)
    #expect(vm.recipe == nil)
  }

  @Test func addToGroceryListWorksFromOverview() {
    let vm = makeCatalogVM()
    vm.apply(.selectRecipe(name: "cookies"))
    vm.apply(.addToGroceryList(item: "flour"))
    #expect(vm.groceryList == ["flour"])
    #expect(vm.phase == .overview)
  }

  @Test func showGroceryListWorksFromOverview() {
    let vm = makeCatalogVM()
    vm.apply(.selectRecipe(name: "cookies"))
    vm.apply(.showGroceryList)
    #expect(vm.groceryOverlayVisible == true)
  }
}
