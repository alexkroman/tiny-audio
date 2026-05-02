import SwiftUI

struct TranscriptList: View {
  let transcripts: [String]
  let emptyPrompt: String
  @Environment(\.accessibilityReduceMotion) private var reduceMotion

  var body: some View {
    if transcripts.isEmpty {
      EmptyState(prompt: emptyPrompt)
    } else {
      ScrollViewReader { proxy in
        ScrollView {
          LazyVStack(alignment: .leading, spacing: 12) {
            ForEach(Array(transcripts.enumerated()), id: \.offset) { index, text in
              TranscriptBubble(text: text)
                .id(index)
            }
          }
          .padding(.horizontal, 16)
          .padding(.vertical, 12)
          .frame(maxWidth: 640, alignment: .leading)
          .frame(maxWidth: .infinity)
        }
        .onChange(of: transcripts.count) { _, newCount in
          guard newCount > 0 else { return }
          let lastIndex = newCount - 1
          if reduceMotion {
            proxy.scrollTo(lastIndex, anchor: .bottom)
          } else {
            withAnimation(.smooth) {
              proxy.scrollTo(lastIndex, anchor: .bottom)
            }
          }
        }
      }
    }
  }
}

#Preview("Empty") {
  TranscriptList(transcripts: [], emptyPrompt: "Tap Record to start listening.")
}

#Preview("With content") {
  TranscriptList(
    transcripts: [
      "The first finalized utterance.",
      "A second one to verify spacing.",
    ],
    emptyPrompt: "Tap Record to start listening."
  )
}
