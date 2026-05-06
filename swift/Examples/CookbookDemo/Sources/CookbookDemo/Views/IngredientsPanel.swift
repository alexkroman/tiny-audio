import SwiftUI

struct IngredientsPanel: View {
  let ingredients: [String]

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Ingredients").font(.title2.weight(.semibold))
      Divider()
      ForEach(ingredients, id: \.self) { item in
        Text("• \(item)").font(.body)
      }
      Spacer()
    }
    .padding(24)
    .frame(width: 320)
    .frame(maxHeight: .infinity)
    .background(.thinMaterial)
  }
}
