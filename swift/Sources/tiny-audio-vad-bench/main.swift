// swift/Sources/tiny-audio-vad-bench/main.swift
//
// Per-frame latency benchmark for SileroVAD across CoreML compute units.
// Standalone executable that measures one compute-units choice per
// invocation — releasing the underlying CoreML MLModel between two
// measurements crashes inside its deinit on macOS, so each config gets
// its own process.
//
// Run with:
//   swift run --package-path swift -c release tiny-audio-vad-bench cpuOnly
//   swift run --package-path swift -c release tiny-audio-vad-bench cpuAndNeuralEngine

import CoreML
import Foundation

@_spi(Bench) import TinyAudio

@main
struct VADBench {
  static func main() throws {
    let warmup = 50
    let measured = 1_000

    let label = CommandLine.arguments.dropFirst().first ?? "cpuAndNeuralEngine"
    let units: MLComputeUnits
    switch label {
    case "cpuOnly":
      units = .cpuOnly
    case "cpuAndNeuralEngine":
      units = .cpuAndNeuralEngine
    case "cpuAndGPU":
      units = .cpuAndGPU
    case "all":
      units = .all
    default:
      print(
        "usage: tiny-audio-vad-bench <cpuOnly|cpuAndNeuralEngine|cpuAndGPU|all>")
      exit(2)
    }

    var speech = [Float](repeating: 0, count: SileroVAD.frameSize)
    for i in 0..<speech.count {
      let t = Float(i) / 16_000
      speech[i] = 0.4 * sin(2 * .pi * 200 * t) * (0.5 + 0.5 * sin(2 * .pi * 5 * t))
    }
    let speechSlice = speech[...]

    let vad = try SileroVAD(computeUnits: units)

    for _ in 0..<warmup {
      _ = try vad.process(speechSlice)
    }

    var samplesNs: [UInt64] = []
    samplesNs.reserveCapacity(measured)
    for _ in 0..<measured {
      let start = DispatchTime.now()
      _ = try vad.process(speechSlice)
      let end = DispatchTime.now()
      samplesNs.append(end.uptimeNanoseconds - start.uptimeNanoseconds)
    }

    samplesNs.sort()
    let n = samplesNs.count
    let totalNs = samplesNs.reduce(UInt64(0), +)
    let avgMs = Double(totalNs) / Double(n) / 1_000_000
    let p50Ms = Double(samplesNs[n / 2]) / 1_000_000
    let p99Index = min(n - 1, Int(Double(n) * 0.99))
    let p99Ms = Double(samplesNs[p99Index]) / 1_000_000

    print(
      String(
        format: "%-22s  avg=%6.3f ms   p50=%6.3f ms   p99=%6.3f ms",
        label, avgMs, p50Ms, p99Ms))
    fflush(stdout)

    // MLModel deinit on a heavily-used model can crash on macOS — exit
    // cleanly before ARC tears it down.
    exit(0)
  }
}
