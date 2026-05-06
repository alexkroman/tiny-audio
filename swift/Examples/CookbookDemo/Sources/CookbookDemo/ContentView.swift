import SwiftUI
import TinyAudio

struct ContentView: View {
  @State private var vm: RecipeViewModel? = nil
  @State private var loadProgress: Double = 0.0
  @Environment(\.scenePhase) private var scenePhase

  // Hold strong refs across view rebuilds.
  @State private var transcriber: Transcriber? = nil
  @State private var mic: MicrophoneTranscriber? = nil
  @State private var pipeline: CommandPipeline? = nil
  @State private var timerController: TimerController? = nil
  @State private var consumeTask: Task<Void, Never>? = nil

  var body: some View {
    Group {
      if let vm {
        switch vm.phase {
        case .loading:
          LoadingView(progress: loadProgress)
        case .cooking:
          CookingView(vm: vm)
        case .micDenied:
          ErrorView(
            title: "Microphone access required",
            message:
              "Cookbook needs the microphone to hear cooking commands. Grant access in System Settings → Privacy & Security → Microphone, then relaunch.",
            hint: "Press ⌘Q to quit."
          )
        case .modelFailed(let msg):
          ErrorView(
            title: "Couldn't load the speech model",
            message: msg,
            hint: "Press ⌘R to retry, ⌘Q to quit."
          )
        }
      } else {
        LoadingView(progress: loadProgress)
      }
    }
    .task { await bootstrap() }
    .onChange(of: scenePhase) { _, phase in
      if phase != .active { Task { await stopMic() } }
    }
  }

  private func bootstrap() async {
    do {
      let recipe = try Recipe.bundled()
      let vm = await MainActor.run { RecipeViewModel(recipe: recipe) }
      self.vm = vm

      let t = try await Transcriber.load()
      self.transcriber = t
      let session = await t.makeChatSession()
      let classifier = LLMIntentClassifier(session: session)
      let pipeline = CommandPipeline(viewModel: vm, classifier: classifier)
      self.pipeline = pipeline

      let m = try MicrophoneTranscriber(transcriber: t)
      self.mic = m
      try await m.start()

      let timerCtl = await MainActor.run { TimerController(viewModel: vm) }
      await MainActor.run { timerCtl.start() }
      self.timerController = timerCtl

      consumeTask = Task { [pipeline, m] in
        await pipeline.consume(events: m.events)
      }

      await MainActor.run { vm.phase = .cooking }
    } catch TinyAudioError.micPermissionDenied {
      await MainActor.run { vm?.phase = .micDenied }
    } catch {
      let msg = String(describing: error)
      await MainActor.run { vm?.phase = .modelFailed(msg) }
    }
  }

  private func stopMic() async {
    consumeTask?.cancel()
    consumeTask = nil
    await mic?.stop()
    mic = nil
    await MainActor.run { timerController?.stop() }
  }
}
