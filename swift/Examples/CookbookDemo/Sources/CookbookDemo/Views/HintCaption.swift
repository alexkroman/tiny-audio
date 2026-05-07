import SwiftUI

struct HintCaption: View {
  let text: String

  var body: some View {
    HStack {
      Text(text)
        .font(.callout)
        .foregroundStyle(.secondary)
      Spacer()
    }
    .padding(.horizontal, 24).padding(.top, 12)
  }
}
