import SwiftUI

struct GroceryBadge: View {
  let count: Int
  var body: some View {
    HStack(spacing: 4) {
      Image(systemName: "cart.fill")
      Text("\(count)").monospacedDigit()
    }
    .font(.callout)
    .padding(.horizontal, 10).padding(.vertical, 4)
    .background(
      Capsule().fill(Color.secondary.opacity(0.15))
    )
  }
}
