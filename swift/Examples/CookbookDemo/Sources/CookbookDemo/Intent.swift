import Foundation

enum Intent: Equatable, Sendable {
  case nextStep
  case previousStep
  case repeatStep
  case restart
  case readIngredients
  case setTimer(seconds: Int)
  case cancelTimer
  case addToGroceryList(item: String)
  case showGroceryList
  case none
}

extension Intent {
  /// Parse the LLM's output into an Intent. Any failure (missing slot,
  /// unknown intent name, malformed JSON, prose wrapper) collapses to `.none`.
  static func from(json: String) -> Intent {
    guard let data = extractFirstJSONObject(json)?.data(using: .utf8) else { return .none }
    guard let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      return .none
    }
    guard let name = dict["intent"] as? String else { return .none }
    switch name {
    case "next_step": return .nextStep
    case "previous_step": return .previousStep
    case "repeat_step": return .repeatStep
    case "restart": return .restart
    case "read_ingredients": return .readIngredients
    case "set_timer":
      guard let s = dict["seconds"] as? Int, s > 0 else { return .none }
      return .setTimer(seconds: s)
    case "cancel_timer": return .cancelTimer
    case "add_to_grocery_list":
      guard let item = (dict["item"] as? String)?
        .trimmingCharacters(in: .whitespacesAndNewlines),
        !item.isEmpty
      else { return .none }
      return .addToGroceryList(item: item)
    case "show_grocery_list": return .showGroceryList
    case "none": return .none
    default: return .none
    }
  }

  /// Find the first balanced `{...}` substring. Tolerates prose preambles.
  private static func extractFirstJSONObject(_ text: String) -> String? {
    guard let start = text.firstIndex(of: "{") else { return nil }
    var depth = 0
    var i = start
    while i < text.endIndex {
      let ch = text[i]
      if ch == "{" { depth += 1 }
      if ch == "}" {
        depth -= 1
        if depth == 0 {
          return String(text[start...i])
        }
      }
      i = text.index(after: i)
    }
    return nil
  }
}
