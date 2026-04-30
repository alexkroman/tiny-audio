// swift/Tests/TinyAudioTests/FixtureLoader.swift
import Foundation

/// Helpers for loading binary fixture files dumped by `scripts/dump_swift_fixtures.py`.
enum FixtureLoader {
    /// Load a raw Float32 fixture file (`reference_*.bin`) and return the flat
    /// array of values. Use `shape(of:)` to retrieve the expected shape.
    static func loadFloat32(name: String) throws -> [Float] {
        guard let url = Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures") else {
            throw NSError(domain: "FixtureLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "missing fixture: \(name)"])
        }
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.size
        return data.withUnsafeBytes { raw in
            Array(UnsafeBufferPointer(start: raw.baseAddress!.assumingMemoryBound(to: Float.self), count: count))
        }
    }

    static func loadShapes() throws -> [String: [String: Any]] {
        guard let url = Bundle.module.url(forResource: "shapes", withExtension: "json", subdirectory: "Fixtures") else {
            return [:]
        }
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: [String: Any]]
        return json
    }

    static func shape(of name: String) throws -> [Int] {
        let shapes = try loadShapes()
        guard let entry = shapes[name], let s = entry["shape"] as? [Int] else {
            throw NSError(domain: "FixtureLoader", code: 2, userInfo: [NSLocalizedDescriptionKey: "no shape for \(name)"])
        }
        return s
    }
}
