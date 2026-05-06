import SwiftUI

struct CookingView: View {
  let vm: RecipeViewModel

  var body: some View {
    ZStack(alignment: .topTrailing) {
      mainColumn
      if vm.ingredientsVisible {
        HStack { Spacer(); IngredientsPanel(ingredients: vm.recipe.ingredients) }
          .transition(.move(edge: .trailing))
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

  private var mainColumn: some View {
    VStack(spacing: 0) {
      topBar
      Divider()
      StepCard(
        stepNumber: vm.currentStepIndex + 1,
        totalSteps: vm.recipe.steps.count,
        stepText: vm.recipe.steps[vm.currentStepIndex]
      )
      Divider()
      heardCaption
    }
  }

  private var topBar: some View {
    HStack(spacing: 16) {
      Text(vm.recipe.title).font(.title2.weight(.semibold))
      Spacer()
      ListeningIndicator(state: vm.listeningState)
      if vm.groceryList.count > 0 {
        GroceryBadge(count: vm.groceryList.count)
      }
      if let t = vm.timer {
        TimerChip(secondsRemaining: t.secondsRemaining)
      }
    }
    .padding(.horizontal, 24).padding(.vertical, 14)
  }

  private var heardCaption: some View {
    HStack {
      Text(vm.lastHeardText.isEmpty ? " " : "heard: \"\(vm.lastHeardText)\"")
        .font(.callout)
        .foregroundStyle(.tertiary)
      Spacer()
    }
    .padding(.horizontal, 24).padding(.vertical, 12)
  }
}
