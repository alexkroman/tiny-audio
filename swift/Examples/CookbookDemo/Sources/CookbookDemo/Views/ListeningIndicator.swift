import SwiftUI

struct ListeningIndicator: View {
  let state: RecipeViewModel.ListeningState

  var body: some View {
    HStack(spacing: 8) {
      Group {
        switch state {
        case .idle:
          Circle().fill(Color.secondary.opacity(0.4))
            .frame(width: 10, height: 10)
        case .hearing:
          Circle().stroke(Color.green, lineWidth: 2)
            .frame(width: 14, height: 14)
            .scaleEffect(1.0)
            .animation(
              .easeInOut(duration: 0.6).repeatForever(autoreverses: true),
              value: state)
        case .thinking:
          ProgressView().controlSize(.small)
        }
      }
      Text(label).font(.caption).foregroundStyle(.secondary)
    }
  }

  private var label: String {
    switch state {
    case .idle: return "listening"
    case .hearing: return "hearing"
    case .thinking: return "thinking"
    }
  }
}
