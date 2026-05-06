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
}
