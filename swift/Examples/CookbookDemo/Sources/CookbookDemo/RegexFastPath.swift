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
      return nil
    }
  }
}
