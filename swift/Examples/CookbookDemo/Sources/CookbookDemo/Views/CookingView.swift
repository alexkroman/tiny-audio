import SwiftUI

struct CookingView: View {
  let vm: RecipeViewModel

  var body: some View {
    ZStack(alignment: .topTrailing) {
      if let recipe = vm.recipe {
        mainColumn(recipe: recipe)
        if vm.ingredientsVisible {
          IngredientsPanel(ingredients: recipe.ingredients)
            .transition(.move(edge: .trailing))
        }
      }
      if vm.groceryOverlayVisible {
        GroceryListOverlay(items: vm.groceryList)
          .transition(.opacity)
      }
      if vm.recipeComplete {
        VStack(spacing: 12) {
          Text("Recipe complete").font(.system(size: 40, weight: .semibold))
          Text("Say \"restart\" to begin again.").foregroundStyle(.secondary)
        }
        .padding(40)
        .background(.regularMaterial)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .transition(.opacity)
      }
    }
    .animation(.easeInOut(duration: 0.18), value: vm.ingredientsVisible)
    .animation(.easeInOut(duration: 0.18), value: vm.groceryOverlayVisible)
    .animation(.easeInOut(duration: 0.18), value: vm.recipeComplete)
  }

  private func mainColumn(recipe: Recipe) -> some View {
    VStack(spacing: 0) {
      ScreenTopBar(
        title: recipe.title,
        listeningState: vm.listeningState,
        groceryCount: vm.groceryList.count
      ) {
        if let t = vm.timer {
          TimerChip(secondsRemaining: t.secondsRemaining)
        }
      }
      Divider()
      StepCard(
        stepNumber: vm.currentStepIndex + 1,
        totalSteps: recipe.steps.count,
        stepText: recipe.steps[vm.currentStepIndex]
      )
      Divider()
      HeardCaption(text: vm.lastHeardText)
    }
  }
}
