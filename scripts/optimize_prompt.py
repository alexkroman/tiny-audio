#!/usr/bin/env python3
"""
Prompt optimization script to find the best prompt for lowest WER.
Runs evaluation 50 times with different prompt variations.
"""

import json
import re
import subprocess
from datetime import datetime

# Define 50 different prompt variations to test
PROMPT_VARIATIONS = [
    # Direct/Simple variations
    "Translate the following audio to English: <audio>",
    "Transcribe this audio to English: <audio>",
    "Transcribe: <audio>",
    "Translate: <audio>",
    "<audio>",
    "Repeat the following text, without any explanation: <audio>",
    # Instructional variations
    "Please translate the following audio to English: <audio>",
    "Please transcribe this audio to English: <audio>",
    "Carefully transcribe the audio to English: <audio>",
    "Accurately translate this audio to English: <audio>",
    "Listen carefully and transcribe to English: <audio>",
    # With context about task
    "Audio transcription task - translate to English: <audio>",
    "Speech-to-text: Translate the audio to English: <audio>",
    "Transcription: Convert this audio to English: <audio>",
    "ASR task: Transcribe the audio to English: <audio>",
    "Audio-to-text translation to English: <audio>",
    # Emphasizing accuracy
    "Transcribe the audio precisely to English: <audio>",
    "Provide exact English transcription: <audio>",
    "Generate accurate English transcription: <audio>",
    "Word-for-word English transcription: <audio>",
    "Precise English translation of audio: <audio>",
    # Different phrasing
    "What is said in this audio (in English)? <audio>",
    "Write down what is said in English: <audio>",
    "Convert speech to English text: <audio>",
    "Generate English text from audio: <audio>",
    "Output English transcription: <audio>",
    # With formatting hints
    "Transcribe to English (text only): <audio>",
    "English transcription without punctuation: <audio>",
    "Plain English transcription: <audio>",
    "English text from audio: <audio>",
    "Audio content in English: <audio>",
    # Action-oriented
    "Transcribe: <audio>",
    "Translate to English: <audio>",
    "Convert to English: <audio>",
    "To English: <audio>",
    "English: <audio>",
    # Question format
    "What does the audio say in English? <audio>",
    "What is being said in English? <audio>",
    "What is the English transcription? <audio>",
    "What is spoken in this audio (English)? <audio>",
    "What are the English words in this audio? <audio>",
    # Natural language
    "Listen to the audio and write it in English: <audio>",
    "Hear the audio and transcribe to English: <audio>",
    "Process this audio and output English: <audio>",
    "Analyze the audio and provide English text: <audio>",
    "Decode the audio to English: <audio>",
    # Minimal
    "<audio>",
    "Transcribe <audio>",
    "English transcription <audio>",
    "Translate <audio>",
    "Audio: <audio>",
]


def run_evaluation(prompt: str, run_number: int) -> dict:
    """Run evaluation with a specific prompt and return results."""
    print(f"\n{'=' * 80}")
    print(f"Run {run_number}/50")
    print(f"Testing prompt: {prompt}")
    print(f"{'=' * 80}\n")

    cmd = [
        "poetry",
        "run",
        "eval",
        "mazesmazes/tiny-audio",
        "--dataset",
        "loquacious",
        "--max-samples",
        "25",
        "--user-prompt",
        prompt,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per run
        )

        output = result.stdout + result.stderr

        # Extract WER from output
        wer_match = re.search(r"WER[:\s]+([0-9.]+)", output, re.IGNORECASE)
        wer = float(wer_match.group(1)) if wer_match else None

        # Try to extract other metrics
        cer_match = re.search(r"CER[:\s]+([0-9.]+)", output, re.IGNORECASE)
        cer = float(cer_match.group(1)) if cer_match else None

        return {
            "run": run_number,
            "prompt": prompt,
            "wer": wer,
            "cer": cer,
            "success": wer is not None,
            "output": output[-1000:],  # Last 1000 chars
            "error": None,
        }

    except subprocess.TimeoutExpired:
        return {
            "run": run_number,
            "prompt": prompt,
            "wer": None,
            "cer": None,
            "success": False,
            "output": "",
            "error": "Timeout",
        }
    except Exception as e:
        return {
            "run": run_number,
            "prompt": prompt,
            "wer": None,
            "cer": None,
            "success": False,
            "output": "",
            "error": str(e),
        }


def main():
    """Main execution function."""
    from datetime import timezone

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = f"prompt_optimization_results_{timestamp}.json"

    all_results = []

    print("Starting prompt optimization experiment")
    print(f"Total variations to test: {len(PROMPT_VARIATIONS)}")
    print(f"Results will be saved to: {results_file}")

    for i, prompt in enumerate(PROMPT_VARIATIONS, 1):
        result = run_evaluation(prompt, i)
        all_results.append(result)

        # Print immediate result
        if result["success"]:
            print(f"✓ WER: {result['wer']:.4f}")
            if result["cer"]:
                print(f"  CER: {result['cer']:.4f}")
        else:
            print(f"✗ Failed: {result['error']}")

        # Save intermediate results
        from pathlib import Path

        with Path(results_file).open("w") as f:
            json.dump(all_results, f, indent=2)

    # Analyze results
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}\n")

    successful_results = [r for r in all_results if r["success"]]

    if successful_results:
        # Sort by WER
        sorted_results = sorted(successful_results, key=lambda x: x["wer"])

        print(f"Successful runs: {len(successful_results)}/{len(PROMPT_VARIATIONS)}\n")

        print("TOP 10 BEST PROMPTS (by WER):")
        print("-" * 80)
        for i, result in enumerate(sorted_results[:10], 1):
            print(f"\n{i}. WER: {result['wer']:.4f}")
            if result["cer"]:
                print(f"   CER: {result['cer']:.4f}")
            print(f"   Prompt: {result['prompt']}")

        print(f"\n{'=' * 80}")
        print("BEST PROMPT:")
        print(f"{'=' * 80}")
        best = sorted_results[0]
        print(f"WER: {best['wer']:.4f}")
        if best["cer"]:
            print(f"CER: {best['cer']:.4f}")
        print(f"Prompt: {best['prompt']}")

        print(f"\n{'=' * 80}")
        print("WORST PROMPT:")
        print(f"{'=' * 80}")
        worst = sorted_results[-1]
        print(f"WER: {worst['wer']:.4f}")
        if worst["cer"]:
            print(f"CER: {worst['cer']:.4f}")
        print(f"Prompt: {worst['prompt']}")

        # Statistics
        wers = [r["wer"] for r in successful_results]
        avg_wer = sum(wers) / len(wers)
        print(f"\n{'=' * 80}")
        print("STATISTICS:")
        print(f"{'=' * 80}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Best WER: {min(wers):.4f}")
        print(f"Worst WER: {max(wers):.4f}")
        print(f"WER Range: {max(wers) - min(wers):.4f}")
    else:
        print("No successful runs!")

    print(f"\nFull results saved to: {results_file}")


if __name__ == "__main__":
    main()
