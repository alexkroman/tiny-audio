import SwiftUI

struct TranscriptBubble: View {
  let text: String

  var body: some View {
    Text(text)
      .font(.body)
      .multilineTextAlignment(.leading)
      .frame(maxWidth: .infinity, alignment: .leading)
      .padding(12)
      .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
      .accessibilityElement(children: .combine)
      .contextMenu {
        Button {
          copyToPasteboard(text)
        } label: {
          Label("Copy", systemImage: "doc.on.doc")
        }
      }
      .textSelection(.enabled)
  }

  private func copyToPasteboard(_ string: String) {
    #if os(iOS)
      UIPasteboard.general.string = string
    #elseif os(macOS)
      NSPasteboard.general.clearContents()
      NSPasteboard.general.setString(string, forType: .string)
    #endif
  }
}

#Preview {
  TranscriptBubble(text: "This is a sample finalized transcript bubble.")
    .padding()
}
