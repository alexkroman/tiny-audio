import SwiftUI

struct LoadingView: View {
  let progress: Double  // 0.0...1.0; if NaN, treat as indeterminate

  var body: some View {
    VStack(spacing: 24) {
      Text("Cookbook")
        .font(.system(size: 56, weight: .semibold, design: .rounded))
      ProgressView(value: progress.isFinite ? progress : nil, total: 1.0)
        .progressViewStyle(.linear)
        .frame(maxWidth: 360)
      Text("Loading on-device speech model…")
        .font(.callout)
        .foregroundStyle(.secondary)
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background(Color.black.opacity(0.05))
  }
}
