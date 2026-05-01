// swift/Sources/tiny-audio-swift-eval/main.swift
//
// Tiny CLI binary used by `ta eval -m swift://...` — loads the SDK's
// Transcriber once, then processes file paths from stdin (one per line),
// emitting JSON results to stdout (one per line).
//
// Protocol:
//   Startup: emit `{"ready": true}` to stdout once load + warmup is done.
//             On startup failure, emit `{"error": "<msg>"}` and exit nonzero.
//   Per request: read a line from stdin = absolute file path.
//             Emit `{"text": "<transcript>", "elapsed_ms": N}` to stdout.
//             On error, emit `{"error": "<msg>"}` (don't exit; keep accepting
//             requests).
//   Shutdown: stdin EOF -> graceful exit 0.

import Foundation
import TinyAudio

// MARK: - Codable message types

private struct ReadyMsg: Encodable { let ready: Bool }
private struct ResultMsg: Encodable {
  let text: String
  let elapsedMs: Int

  enum CodingKeys: String, CodingKey {
    case text
    case elapsedMs = "elapsed_ms"
  }
}
private struct ErrorMsg: Encodable {
  let error: String
  let path: String?
}

@main
struct TinyAudioEvalCLI {
  static func main() async {
    // Ensure mlx.metallib is next to the executable before any MLX use.
    MLXBootstrap.ensureMetallibAvailable()

    // Parse args: --repo <repo-id> | --local <dir>. Default: .defaultHub.
    let args = Array(CommandLine.arguments.dropFirst())
    let source: WeightSource = parseSource(args)

    let transcriber: Transcriber
    do {
      transcriber = try await Transcriber.load(from: source, progress: nil)
    } catch {
      emit(ErrorMsg(error: "load failed: \(error)", path: nil))
      exit(1)
    }

    emit(ReadyMsg(ready: true))

    // Read paths from stdin until EOF.
    while let line = readLine() {
      let path = line.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !path.isEmpty else { continue }
      guard !path.contains("\n") else {
        emit(ErrorMsg(error: "path contains newline", path: nil))
        continue
      }
      let url = URL(fileURLWithPath: path)
      let started = ContinuousClock.now
      do {
        let text = try await transcriber.transcribe(.file(url))
        let elapsed = started.duration(to: ContinuousClock.now)
        let ns =
          elapsed.components.seconds * 1_000_000_000 + elapsed.components.attoseconds
          / 1_000_000_000
        let elapsedMs = Int(ns / 1_000_000)
        emit(ResultMsg(text: text, elapsedMs: elapsedMs))
      } catch {
        emit(ErrorMsg(error: "\(error)", path: path))
      }
    }
  }

  private static func parseSource(_ args: [String]) -> WeightSource {
    var i = 0
    while i < args.count {
      switch args[i] {
      case "--repo":
        if i + 1 < args.count {
          let repo = args[i + 1]
          return .hub(repoID: repo, revision: nil)
        }
        return .defaultHub
      case "--local":
        if i + 1 < args.count {
          return .localDirectory(URL(fileURLWithPath: args[i + 1]))
        }
        return .defaultHub
      default:
        i += 1
        continue
      }
    }
    return .defaultHub
  }

  private static func emit<T: Encodable>(_ msg: T) {
    do {
      var data = try JSONEncoder().encode(msg)
      data.append(0x0a)  // '\n'
      FileHandle.standardOutput.write(data)
    } catch {
      FileHandle.standardError.write(Data("emit failed: \(error)\n".utf8))
    }
  }
}
