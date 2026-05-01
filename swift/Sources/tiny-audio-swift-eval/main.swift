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

@main
struct TinyAudioEvalCLI {
    static func main() async {
        // Parse args: --repo <repo-id> | --local <dir>. Default: .defaultHub.
        let args = Array(CommandLine.arguments.dropFirst())
        let source: WeightSource = parseSource(args)

        let transcriber: Transcriber
        do {
            transcriber = try await Transcriber.load(from: source, progress: nil)
        } catch {
            emitError("load failed: \(error)")
            exit(1)
        }

        emit(["ready": true])

        // Read paths from stdin until EOF.
        while let line = readLine() {
            let path = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !path.isEmpty else { continue }
            let url = URL(fileURLWithPath: path)
            let started = Date()
            do {
                let text = try await transcriber.transcribe(.file(url))
                let elapsedMs = Int(Date().timeIntervalSince(started) * 1000)
                emit(["text": text, "elapsed_ms": elapsedMs])
            } catch {
                emit(["error": "\(error)", "path": path])
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

    private static func emit(_ object: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            FileHandle.standardError.write(Data("emit failed for \(object)\n".utf8))
            return
        }
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    }

    private static func emitError(_ msg: String) {
        emit(["error": msg])
    }
}
