# CookbookDemo

A hands-free, voice-controlled cooking demo for macOS. Walks the user through a single
bundled recipe via spoken commands. No mouse interaction during cooking — the only
mouse touch is the macOS microphone-permission prompt on first launch.

## Requirements

- macOS 14+
- Apple Silicon recommended
- A microphone

## Run

```
swift run --package-path swift/Examples/CookbookDemo CookbookDemo
```

## What it shows

- TinyAudio's `Transcriber` + `MicrophoneTranscriber` for live mic transcription.
- `Transcriber.makeChatSession()` reusing the bundled Qwen3-0.6B as a structured
  intent classifier — no second model on disk.
- A `CommandPipeline` actor that ingests VAD-segmented utterances, runs a regex
  fast-path for unambiguous literals, falls back to the LLM for everything else,
  and dispatches resolved intents to a SwiftUI view model on the main actor.

## Voice commands

| Say | Effect |
| ---------------------------------- | ----------------------- |
| "next" / "next step" / "go on" | advance one step |
| "back" / "go back" / "previous" | go back one step |
| "repeat" / "say that again" | re-display current step |
| "restart" / "start over" | jump to step 1 |
| "what are the ingredients" | show ingredients panel |
| "set a timer for five minutes" | start a timer |
| "cancel timer" / "stop the timer" | clear active timer |
| "add olive oil to my list" | add to grocery list |
| "show my grocery list" | full-screen list overlay |

## Manual demo checklist

Run through this end-to-end before any live demo:

1. Launch — model loads, mic prompt appears, accept. Cooking view shows step 1 of 8.
1. Say "next" — advances to step 2.
1. Say "go back" — returns to step 1.
1. Say "set a timer for thirty seconds" — chip appears counting down.
1. Wait 5 seconds, say "cancel timer" — chip disappears.
1. Say "what are the ingredients" — panel slides in.
1. Say "next" — panel dismisses, step advances.
1. Say "add olive oil to my list" — badge "🛒 1" appears in top bar.
1. Say "show my grocery list" — full-screen overlay.
1. Say "back" — overlay dismisses.
1. Speak unrelated chatter ("the dog is barking") — last-heard caption updates,
   no state change.
1. Restart → speak "next" 8 times — "Recipe complete" overlay appears.
1. Say "restart" — overlay dismisses, back at step 1.

## LLM fixture test

To verify intent-classification quality on a 25-utterance fixture:

```
COOKBOOK_LLM_TEST=1 swift test --package-path swift/Examples/CookbookDemo \
  --filter LLMIntentClassifierFixtureTests
```

Threshold: ≥90% match.
