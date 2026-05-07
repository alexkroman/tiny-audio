import SwiftUI

struct RecipeOverviewView: View {
  let vm: RecipeViewModel

  var body: some View {
    ZStack {
      mainColumn
      if vm.groceryOverlayVisible {
        GroceryListOverlay(items: vm.groceryList)
          .transition(.opacity)
      }
    }
    .animation(.easeInOut(duration: 0.18), value: vm.groceryOverlayVisible)
  }

  private var mainColumn: some View {
    VStack(spacing: 0) {
      ScreenTopBar(
        title: vm.recipe?.title ?? "Recipe",
        listeningState: vm.listeningState,
        groceryCount: vm.groceryList.count
      )
      Divider()
      if let recipe = vm.recipe {
        ScrollView {
          VStack(alignment: .leading, spacing: 12) {
            Text("Ingredients").font(.title3.weight(.semibold))
            ForEach(recipe.ingredients, id: \.self) { item in
              Text("• \(item)").font(.body)
            }
          }
          .frame(maxWidth: .infinity, alignment: .leading)
          .padding(24)
        }
      } else {
        Spacer()
      }
      Divider()
      HintCaption(
        text:
          "Say \"add <item> to grocery list\" to save ingredients, \"next\" to start cooking, "
          + "or \"back\" to pick a different recipe."
      )
      HeardCaption(text: vm.lastHeardText)
    }
  }
}
