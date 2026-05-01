import SwiftUI
import TinyAudio

@main
struct TinyAudioDemoApp: App {
    init() {
        MLXBootstrap.ensureMetallibAvailable()
    }

    var body: some Scene {
        WindowGroup("TinyAudio Demo") {
            ContentView()
                .frame(minWidth: 480, minHeight: 320)
        }
    }
}
