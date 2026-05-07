import Foundation
import Testing

@testable import CookbookDemo

@Suite("RegexFastPath")
struct RegexFastPathTests {
  @Test func matchesNext() { #expect(RegexFastPath.match("next") == .nextStep) }
  @Test func matchesNextWithPunctuation() { #expect(RegexFastPath.match("Next.") == .nextStep) }
  @Test func matchesBack() { #expect(RegexFastPath.match("back") == .previousStep) }
  @Test func matchesPrevious() { #expect(RegexFastPath.match("previous") == .previousStep) }
  @Test func matchesRepeat() { #expect(RegexFastPath.match("repeat") == .repeatStep) }
  @Test func matchesRestart() { #expect(RegexFastPath.match("restart") == .restart) }
  @Test func matchesCancelTimer() { #expect(RegexFastPath.match("cancel timer") == .cancelTimer) }
  @Test func matchesCancelTheTimer() {
    #expect(RegexFastPath.match("cancel the timer") == .cancelTimer)
  }

  @Test func nextlyDoesNotMatch() { #expect(RegexFastPath.match("nextly") == nil) }
  @Test func sentenceContainingNextDoesNotMatch() {
    // The whole point: don't trip on speech that merely contains a keyword.
    #expect(RegexFastPath.match("the next ingredient is flour") == nil)
  }
  @Test func emptyStringIsNil() { #expect(RegexFastPath.match("") == nil) }

  @Test func matchesAddBakingPowderToGroceryList() {
    #expect(
      RegexFastPath.match("add baking powder to grocery list")
        == .addToGroceryList(item: "baking powder"))
  }

  @Test func matchesAddOliveOilToMyList() {
    #expect(
      RegexFastPath.match("add olive oil to my list")
        == .addToGroceryList(item: "olive oil"))
  }

  @Test func matchesAddSaltToTheShoppingList() {
    #expect(
      RegexFastPath.match("Add salt to the shopping list.")
        == .addToGroceryList(item: "salt"))
  }

  @Test func matchesPutButterOnMyList() {
    #expect(
      RegexFastPath.match("put butter on my list")
        == .addToGroceryList(item: "butter"))
  }

  @Test func bareAddDoesNotMatch() {
    // No "to/on ... list" anchor — too ambiguous; punt to the LLM.
    #expect(RegexFastPath.match("add salt") == nil)
  }

  @Test func sentenceWithoutListSuffixDoesNotMatch() {
    #expect(RegexFastPath.match("add salt to the soup") == nil)
  }
}
