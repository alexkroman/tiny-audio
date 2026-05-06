import Foundation

@MainActor
final class TimerController {
  private weak var viewModel: RecipeViewModel?
  private var task: Task<Void, Never>?

  init(viewModel: RecipeViewModel) { self.viewModel = viewModel }

  /// Start ticking once. Re-entrant: replaces any prior tick task.
  func start() {
    task?.cancel()
    task = Task { [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(for: .seconds(1))
        guard let vm = self?.viewModel else { return }
        guard var t = vm.timer else { continue }
        let remaining = max(0, Int(t.endsAt.timeIntervalSinceNow.rounded()))
        t.secondsRemaining = remaining
        vm.timer = t
      }
    }
  }

  func stop() {
    task?.cancel()
    task = nil
  }
}
