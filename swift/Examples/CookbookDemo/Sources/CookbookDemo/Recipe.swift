import Foundation

struct Recipe: Codable, Sendable, Equatable {
  let title: String
  let ingredients: [String]
  let steps: [String]
}

extension Recipe {
  /// Load the single recipe bundled into this app target.
  static func bundled() throws -> Recipe {
    guard let url = Bundle.module.url(forResource: "recipe", withExtension: "json") else {
      throw CocoaError(.fileNoSuchFile)
    }
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(Recipe.self, from: data)
  }
}
