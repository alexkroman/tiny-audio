import SwiftUI
import TinyAudio

struct ContentView: View {
  @State private var vm: RecipeViewModel? = nil

  // Hold strong refs across view rebuilds so the model, mic engine, and
  // tick/consume tasks aren't deallocated mid-flight.
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
          LoadingView(progress: .nan)
        case .selecting:
          LoadingView(progress: .nan)
        case .overview:
          LoadingView(progress: .nan)
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
        LoadingView(progress: .nan)
      }
    }
    .task { await bootstrap() }
  }

  private func bootstrap() async {
    do {
      let recipes = try Recipe.bundledAll()
      let vm = RecipeViewModel(recipes: recipes)
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

      let timerCtl = TimerController(viewModel: vm)
      timerCtl.start()
      self.timerController = timerCtl

      consumeTask = Task { [pipeline, m] in
        await pipeline.consume(events: m.events)
      }

      vm.phase = .selecting
    } catch TinyAudioError.micPermissionDenied {
      vm?.phase = .micDenied
    } catch {
      vm?.phase = .modelFailed(String(describing: error))
    }
  }
}
