import SwiftUI

struct TimerChip: View {
  let secondsRemaining: Int

  var body: some View {
    HStack(spacing: 6) {
      Image(systemName: "timer")
      Text(format(secondsRemaining)).monospacedDigit()
    }
    .font(.callout.weight(.medium))
    .padding(.horizontal, 12).padding(.vertical, 6)
    .background(
      RoundedRectangle(cornerRadius: 10)
        .fill(secondsRemaining == 0 ? Color.red.opacity(0.25) : Color.secondary.opacity(0.15))
    )
    .foregroundStyle(secondsRemaining == 0 ? Color.red : Color.primary)
  }

  private func format(_ s: Int) -> String {
    String(format: "%d:%02d", s / 60, s % 60)
  }
}
