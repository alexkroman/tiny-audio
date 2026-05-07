import Foundation
import TinyAudio

actor CommandPipeline {
  private let viewModel: RecipeViewModel
  private let classifier: any IntentClassifying

  init(viewModel: RecipeViewModel, classifier: any IntentClassifying) {
    self.viewModel = viewModel
    self.classifier = classifier
  }

  /// Test-friendly entry point. The production driver calls this from the
  /// MicrophoneTranscriber events loop.
  func handle(transcribedText text: String) async {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }

    await viewModel.setLastHeard(trimmed)
    await viewModel.setListeningState(.thinking)

    let intent: Intent
    if let fast = RegexFastPath.match(trimmed) {
      intent = fast
    } else {
      intent = await classifier.classify(trimmed)
    }

    await viewModel.apply(intent)

    // Fallback for the recipe-selection screen: bare recipe names ("pancakes")
    // often classify as .none because the LLM examples are all phrasal. If
    // we're still in .selecting after applying the classifier's result, try
    // the utterance as a recipe name. matchRecipe silently no-ops on misses.
    if intent == .none, await viewModel.phase == .selecting {
      await viewModel.apply(.selectRecipe(name: trimmed))
    }

    await viewModel.setListeningState(.idle)
  }

  /// Drive the pipeline from a `MicrophoneTranscriber`'s event stream until it
  /// closes. Production code calls this once after `mic.start()`.
  func consume(events: AsyncStream<MicrophoneTranscriber.Event>) async {
    for await event in events {
      switch event {
      case .final(_, let text):
        await viewModel.setListeningState(.hearing)
        await handle(transcribedText: text)
      case .error:
        await viewModel.setListeningState(.idle)
      }
    }
  }
}
