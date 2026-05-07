import Foundation

enum RegexFastPath {
  /// Return an Intent if the utterance is one of a handful of unambiguous,
  /// whole-utterance literals. The match is intentionally strict: the entire
  /// utterance (modulo trailing punctuation) must be the literal command, so
  /// a sentence merely *containing* "next" does not trigger.
  static func match(_ utterance: String) -> Intent? {
    let trimmed =
      utterance
      .lowercased()
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .trimmingCharacters(in: CharacterSet.punctuationCharacters)
    switch trimmed {
    case "next", "next step", "next, please", "next please":
      return .nextStep
    case "back", "go back", "previous", "previous step":
      return .previousStep
    case "repeat", "repeat that", "say that again":
      return .repeatStep
    case "restart", "start over":
      return .restart
    case "cancel timer", "cancel the timer", "stop timer", "stop the timer":
      return .cancelTimer
    default:
      break
    }

    // "add X to (my|the|our)? (grocery|shopping)? list" / "put X on ... list".
    // Captures the item between the verb and the trailing list reference, so
    // the LLM doesn't have to handle this very common phrasing.
    let groceryPattern =
      #/^(?:add|put)\s+(.+?)\s+(?:to|on)\s+(?:(?:the|my|our)\s+)?(?:(?:grocery|shopping)\s+)?list$/#
    if let match = trimmed.wholeMatch(of: groceryPattern) {
      let item = String(match.output.1).trimmingCharacters(in: .whitespacesAndNewlines)
      if !item.isEmpty {
        return .addToGroceryList(item: item)
      }
    }

    return nil
  }
}
