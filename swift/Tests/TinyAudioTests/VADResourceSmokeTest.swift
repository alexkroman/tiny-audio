import Testing
@testable import TinyAudio

@Suite("VADResource")
struct VADResourceSmokeTest {
    @Test func sileroResourceIsBundled() {
        #expect(VADResources.sileroVADURL != nil)
    }
}
