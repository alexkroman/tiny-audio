import Foundation
import Testing

@testable import CookbookDemo

@Suite("Recipe")
struct RecipeTests {
  @Test func bundledAllReturnsFourRecipes() throws {
    let recipes = try Recipe.bundledAll()
    #expect(recipes.count == 4)
    let titles = recipes.map(\.title)
    #expect(titles.contains("Chocolate Chip Cookies"))
    #expect(titles.contains("Pancakes"))
    #expect(titles.contains("Grilled Cheese"))
    #expect(titles.contains("Guacamole"))
  }

  @Test func bundledAllRecipesHaveContent() throws {
    let recipes = try Recipe.bundledAll()
    for recipe in recipes {
      #expect(!recipe.ingredients.isEmpty, "\(recipe.title) ingredients")
      #expect(!recipe.steps.isEmpty, "\(recipe.title) steps")
    }
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
