import SwiftUI

struct EmptyState: View {
  let prompt: String

  var body: some View {
    VStack(spacing: 12) {
      Image(systemName: "waveform")
        .font(.system(size: 48, weight: .light))
        .foregroundStyle(.tertiary)
        .accessibilityHidden(true)
      Text("No transcripts yet")
        .font(.title3)
        .foregroundStyle(.secondary)
      Text(prompt)
        .font(.body)
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .padding()
    .accessibilityElement(children: .combine)
  }
}

#Preview("iOS prompt") {
  EmptyState(prompt: "Tap Record to start listening.")
}

#Preview("macOS prompt") {
  EmptyState(prompt: "Click Record in the toolbar to start listening.")
}
