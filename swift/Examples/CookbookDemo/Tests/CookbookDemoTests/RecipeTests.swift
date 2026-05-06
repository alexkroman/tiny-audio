import Foundation
import Testing

@testable import CookbookDemo

@Suite("Recipe")
struct RecipeTests {
  @Test func decodesFromBundledJSON() throws {
    let recipe = try Recipe.bundled()
    #expect(recipe.title == "Chocolate Chip Cookies")
    #expect(recipe.steps.count == 8)
    #expect(recipe.ingredients.count == 9)
    #expect(recipe.steps.first?.contains("Preheat") == true)
  }

  @Test func decodesFromRawJSON() throws {
    let json = #"""
      {"title":"T","ingredients":["a","b"],"steps":["one","two"]}
      """#.data(using: .utf8)!
    let r = try JSONDecoder().decode(Recipe.self, from: json)
    #expect(r.title == "T")
    #expect(r.ingredients == ["a", "b"])
    #expect(r.steps == ["one", "two"])
  }
}
