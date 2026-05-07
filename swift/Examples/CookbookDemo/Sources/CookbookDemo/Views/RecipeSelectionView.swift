import SwiftUI

struct RecipeSelectionView: View {
  let vm: RecipeViewModel

  var body: some View {
    VStack(spacing: 0) {
      ScreenTopBar(
        title: "Pick a recipe",
        listeningState: vm.listeningState,
        groceryCount: vm.groceryList.count
      )
      Divider()
      ScrollView {
        VStack(spacing: 16) {
          ForEach(vm.recipes, id: \.title) { recipe in
            recipeCard(recipe)
          }
        }
        .padding(24)
      }
      Divider()
      HintCaption(text: "Say a recipe name to begin.")
      HeardCaption(text: vm.lastHeardText)
    }
  }

  private func recipeCard(_ recipe: Recipe) -> some View {
    VStack(alignment: .leading, spacing: 6) {
      Text(recipe.title).font(.title3.weight(.semibold))
      Text("\(recipe.ingredients.count) ingredients · \(recipe.steps.count) steps")
        .font(.callout)
        .foregroundStyle(.secondary)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .padding(20)
    .background(
      RoundedRectangle(cornerRadius: 12).fill(Color.secondary.opacity(0.08))
    )
  }
}
