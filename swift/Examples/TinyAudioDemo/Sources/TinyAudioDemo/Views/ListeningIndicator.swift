import SwiftUI

struct ListeningIndicator: View {
  @Environment(\.accessibilityReduceMotion) private var reduceMotion
  @State private var phase: CGFloat = 0

  private let barCount = 3
  private let barWidth: CGFloat = 3
  private let maxHeight: CGFloat = 18
  private let minHeight: CGFloat = 6
  private let spacing: CGFloat = 3

  var body: some View {
    Group {
      if reduceMotion {
        Circle()
          .fill(Color.red)
          .frame(width: 10, height: 10)
          .accessibilityHidden(true)
      } else {
        HStack(alignment: .center, spacing: spacing) {
          ForEach(0..<barCount, id: \.self) { index in
            Capsule()
              .fill(Color.red)
              .frame(width: barWidth, height: barHeight(for: index))
          }
        }
        .frame(height: maxHeight)
        .onAppear {
          withAnimation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true)) {
            phase = 1
          }
        }
        .accessibilityLabel("Listening")
      }
    }
  }

  private func barHeight(for index: Int) -> CGFloat {
    let stagger = CGFloat(index) * 0.33
    let local = abs((phase + stagger).truncatingRemainder(dividingBy: 1.0) - 0.5) * 2
    return minHeight + (maxHeight - minHeight) * local
  }
}

#Preview {
  ListeningIndicator()
    .padding()
}
