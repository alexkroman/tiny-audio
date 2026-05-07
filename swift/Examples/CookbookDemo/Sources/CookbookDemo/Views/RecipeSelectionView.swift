import SwiftUI

struct RecipeSelectionView: View {
  let vm: RecipeViewModel

  var body: some View {
    VStack(spacing: 0) {
      topBar
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
      hint
      heardCaption
    }
  }

  private var topBar: some View {
    HStack(spacing: 16) {
      Text("Pick a recipe").font(.title2.weight(.semibold))
      Spacer()
      ListeningIndicator(state: vm.listeningState)
      if !vm.groceryList.isEmpty {
        GroceryBadge(count: vm.groceryList.count)
      }
    }
    .padding(.horizontal, 24).padding(.vertical, 14)
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

  private var hint: some View {
    HStack {
      Text("Say a recipe name to begin.")
        .font(.callout)
        .foregroundStyle(.secondary)
      Spacer()
    }
    .padding(.horizontal, 24).padding(.top, 12)
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
