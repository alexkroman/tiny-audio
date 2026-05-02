import SwiftUI
#if os(iOS)
  import UIKit
#elseif os(macOS)
  import AppKit
#endif

struct PermissionBanner: View {
  var body: some View {
    HStack(alignment: .firstTextBaseline, spacing: 12) {
      Image(systemName: "mic.slash.fill")
        .foregroundStyle(.orange)
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 2) {
        Text("Microphone access is required")
          .font(.subheadline.weight(.semibold))
        Text("Grant access to start transcribing.")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Spacer(minLength: 8)
      Button("Open Settings", action: openSettings)
        .buttonStyle(.bordered)
        .controlSize(.small)
    }
    .padding(12)
    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    .accessibilityElement(children: .combine)
  }

  private func openSettings() {
    #if os(iOS)
      if let url = URL(string: UIApplication.openSettingsURLString) {
        UIApplication.shared.open(url)
      }
    #elseif os(macOS)
      if let url = URL(
        string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
      {
        NSWorkspace.shared.open(url)
      }
    #endif
  }
}

#Preview {
  PermissionBanner()
    .padding()
}
