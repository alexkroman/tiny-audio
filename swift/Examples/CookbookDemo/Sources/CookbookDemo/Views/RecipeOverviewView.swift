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
      topBar
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
      hint
      heardCaption
    }
  }

  private var topBar: some View {
    HStack(spacing: 16) {
      Text(vm.recipe?.title ?? "Recipe")
        .font(.title2.weight(.semibold))
      Spacer()
      ListeningIndicator(state: vm.listeningState)
      if !vm.groceryList.isEmpty {
        GroceryBadge(count: vm.groceryList.count)
      }
    }
    .padding(.horizontal, 24).padding(.vertical, 14)
  }

  private var hint: some View {
    HStack {
      Text(
        "Say \"add <item> to grocery list\" to save ingredients, \"next\" to start cooking, "
        + "or \"back\" to pick a different recipe."
      )
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
