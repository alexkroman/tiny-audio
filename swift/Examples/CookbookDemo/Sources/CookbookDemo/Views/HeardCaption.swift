import SwiftUI

struct HeardCaption: View {
  let text: String

  var body: some View {
    HStack {
      Text(text.isEmpty ? " " : "heard: \"\(text)\"")
        .font(.callout)
        .foregroundStyle(.tertiary)
      Spacer()
    }
    .padding(.horizontal, 24).padding(.vertical, 12)
  }
}
