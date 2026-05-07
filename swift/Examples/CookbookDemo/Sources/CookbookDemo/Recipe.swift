import Foundation

struct Recipe: Codable, Sendable, Equatable {
  let title: String
  let ingredients: [String]
  let steps: [String]
}

extension Recipe {
  /// Load all bundled recipes into the app catalog.
  static func bundledAll() throws -> [Recipe] {
    guard let url = Bundle.module.url(forResource: "recipes", withExtension: "json") else {
      throw CocoaError(.fileNoSuchFile)
    }
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode([Recipe].self, from: data)
  }
}
