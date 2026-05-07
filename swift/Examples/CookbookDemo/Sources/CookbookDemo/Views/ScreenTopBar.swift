import SwiftUI

struct ScreenTopBar<Trailing: View>: View {
  let title: String
  let listeningState: RecipeViewModel.ListeningState
  let groceryCount: Int
  @ViewBuilder let trailing: () -> Trailing

  var body: some View {
    HStack(spacing: 16) {
      Text(title).font(.title2.weight(.semibold))
      Spacer()
      ListeningIndicator(state: listeningState)
      if groceryCount > 0 { GroceryBadge(count: groceryCount) }
      trailing()
    }
    .padding(.horizontal, 24).padding(.vertical, 14)
  }
}

extension ScreenTopBar where Trailing == EmptyView {
  init(title: String, listeningState: RecipeViewModel.ListeningState, groceryCount: Int) {
    self.init(
      title: title, listeningState: listeningState, groceryCount: groceryCount
    ) { EmptyView() }
  }
}
