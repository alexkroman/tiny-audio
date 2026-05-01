import SwiftUI

struct RecordButton: View {
  let isListening: Bool
  let isEnabled: Bool
  let action: () -> Void

  var body: some View {
    Button(action: action) {
      Label(label, systemImage: symbol)
        .labelStyle(.titleAndIcon)
        .font(.body.weight(.semibold))
        .frame(minWidth: 44, minHeight: 44)
    }
    .buttonStyle(.borderedProminent)
    .tint(isListening ? .red : .accentColor)
    .controlSize(.large)
    .disabled(!isEnabled)
    .accessibilityLabel(isListening ? "Stop listening" : "Start listening")
    .accessibilityHint("Records audio and transcribes it.")
  }

  private var label: String {
    isListening ? "Stop" : "Record"
  }

  private var symbol: String {
    isListening ? "stop.fill" : "mic.fill"
  }
}

#Preview("Idle") {
  RecordButton(isListening: false, isEnabled: true, action: {})
    .padding()
}

#Preview("Listening") {
  RecordButton(isListening: true, isEnabled: true, action: {})
    .padding()
}
