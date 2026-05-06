import SwiftUI

struct GroceryListOverlay: View {
  let items: [String]

  var body: some View {
    VStack(spacing: 24) {
      Text("Grocery List").font(.system(size: 36, weight: .semibold))
      if items.isEmpty {
        Text("Empty — say \"add olive oil to the list\" to add items.")
          .foregroundStyle(.secondary)
      } else {
        VStack(alignment: .leading, spacing: 8) {
          ForEach(Array(items.enumerated()), id: \.offset) { i, item in
            Text("\(i + 1). \(item)").font(.title3)
          }
        }
      }
      Spacer()
      Text("Say \"back\" or \"next\" to return.")
        .font(.callout).foregroundStyle(.tertiary)
    }
    .padding(40)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background(.regularMaterial)
  }
}
