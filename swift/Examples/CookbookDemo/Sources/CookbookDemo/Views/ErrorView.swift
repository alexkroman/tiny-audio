import SwiftUI

struct ErrorView: View {
  let title: String
  let message: String
  let hint: String

  var body: some View {
    VStack(spacing: 18) {
      Text(title).font(.system(size: 36, weight: .semibold))
      Text(message)
        .multilineTextAlignment(.center)
        .frame(maxWidth: 520)
        .foregroundStyle(.secondary)
      Text(hint).font(.callout).foregroundStyle(.tertiary)
    }
    .padding(40)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}
