import SwiftUI

struct StepCard: View {
  let stepNumber: Int  // 1-based for display
  let totalSteps: Int
  let stepText: String

  var body: some View {
    VStack(spacing: 32) {
      Text("STEP \(stepNumber) OF \(totalSteps)")
        .font(.system(size: 18, weight: .semibold, design: .rounded))
        .tracking(2)
        .foregroundStyle(.secondary)
      Text(stepText)
        .font(.system(size: 40, weight: .regular, design: .serif))
        .multilineTextAlignment(.center)
        .lineSpacing(8)
        .frame(maxWidth: 720)
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .padding(40)
  }
}
