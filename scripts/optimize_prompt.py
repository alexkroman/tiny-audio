"""Optimize AssemblyAI prompts using DSPy MIPROv2 optimizer.

Uses DSPy's MIPROv2 to generate and evaluate candidate prompt instructions
for AssemblyAI's universal-3-pro model, minimizing WER on a held-out dataset.

Usage:
    ASSEMBLYAI_API_KEY=... ANTHROPIC_API_KEY=... poetry run python scripts/optimize_prompt.py \
        --dataset loquacious \
        --opt-samples 100 \
        --auto light
"""

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

os.environ.setdefault("HF_DATASETS_AUDIO_DECODER", "soundfile")

from scripts.eval.audio import TextNormalizer, prepare_wav_bytes
from scripts.eval.datasets import DATASET_REGISTRY, load_eval_dataset

app = typer.Typer(help="Optimize AssemblyAI prompts using DSPy MIPROv2")
console = Console()

SPEECH_MODEL = "universal-3-pro"
normalizer = TextNormalizer()


def collect_samples(dataset_name: str, num_samples: int, split: str = "dev") -> list[dict]:
    """Pre-collect samples from streaming dataset for consistent evaluation."""
    cfg = DATASET_REGISTRY[dataset_name]
    dataset = load_eval_dataset(dataset_name, split)
    samples = []
    for sample in dataset:
        reference = sample[cfg.text_field]
        if isinstance(reference, str) and reference.strip() == "ignore_time_segment_in_scoring":
            continue
        if isinstance(reference, str) and "inaudible" in reference.lower():
            continue
        samples.append(
            {
                "audio": sample[cfg.audio_field],
                "reference": reference,
            }
        )
        if len(samples) % 50 == 0:
            console.print(f"  [dim]Collected {len(samples)} samples...[/dim]")
        if len(samples) >= num_samples:
            break
    return samples


def transcribe_assemblyai(audio, prompt: str, api_key: str) -> str:
    """Transcribe audio using AssemblyAI with the given prompt."""
    import assemblyai as aai

    aai.settings.api_key = api_key
    config_kwargs = {
        "speech_models": [SPEECH_MODEL],
        "language_detection": True,
        "prompt": prompt,
    }
    config = aai.TranscriptionConfig(**config_kwargs)
    transcriber = aai.Transcriber(config=config)
    wav_bytes = prepare_wav_bytes(audio)
    try:
        transcript = transcriber.transcribe(io.BytesIO(wav_bytes))
        return transcript.text or ""
    except Exception as e:
        console.print(f"  [red]AssemblyAI error: {e}[/red]")
        return ""


def wer_metric(example, pred, trace=None):
    """WER-based metric for DSPy (higher = better, 0-1 scale)."""
    import jiwer

    ref = normalizer.normalize(example.reference)
    hyp = normalizer.normalize(pred.transcription)
    if not ref:
        return 1.0
    wer_score = jiwer.wer(ref, hyp)
    return max(0.0, 1.0 - wer_score)


@app.callback(invoke_without_command=True)
def main(
    dataset: str = typer.Option("loquacious", "--dataset", "-d", help="Dataset to evaluate on"),
    split: str = typer.Option(
        "validation",
        "--split",
        "-s",
        help="Dataset split (use validation/dev to avoid overlap with eval test split)",
    ),
    starting_prompt: str = typer.Option(
        "Transcribe the following audio accurately.",
        "--starting-prompt",
        "-p",
        help="Initial prompt instruction",
    ),
    opt_samples: int = typer.Option(100, "--opt-samples", help="Number of samples"),
    auto: str = typer.Option("light", "--auto", help="MIPROv2 preset: light, medium, heavy"),
    num_threads: int = typer.Option(4, "--num-threads", "-w", help="Parallel eval threads"),
    output_dir: str = typer.Option(
        "outputs/prompt_optimization", "--output-dir", "-o", help="Output directory"
    ),
):
    """Optimize AssemblyAI universal-3-pro prompts using DSPy MIPROv2."""
    import dspy
    from dspy.teleprompt import MIPROv2

    api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not api_key:
        console.print("[red]Error: ASSEMBLYAI_API_KEY not set[/red]")
        raise typer.Exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]Error: ANTHROPIC_API_KEY not set[/red]")
        raise typer.Exit(1)

    if dataset not in DATASET_REGISTRY:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        raise typer.Exit(1)

    # Configure DSPy LM (used by MIPROv2 to generate instruction candidates)
    lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929", max_tokens=1024)
    dspy.configure(lm=lm)

    console.print("[bold]AssemblyAI Prompt Optimizer (DSPy MIPROv2)[/bold]")
    console.print(f"  Model: assemblyai/{SPEECH_MODEL}")
    console.print(f"  Dataset: {dataset} (split: {split})")
    console.print(f"  Samples: {opt_samples}")
    console.print(f"  Preset: {auto}")
    console.print(f"  Threads: {num_threads}")
    console.print(f"  Starting prompt: {starting_prompt!r}")

    # Collect samples
    console.print(f"\n[cyan]Collecting {opt_samples} samples from {dataset}/{split}...[/cyan]")
    samples = collect_samples(dataset, opt_samples, split)
    console.print(f"[green]Collected {len(samples)} samples[/green]")

    # Define DSPy module — the instruction on self.predict IS the ASR prompt
    # MIPROv2 optimizes this instruction. The forward method extracts it and
    # passes it as AssemblyAI's `prompt` parameter.
    class ASRModule(dspy.Module):
        """ASR prompt optimization module.

        The instruction on self.predict is used as the AssemblyAI prompt parameter
        for the universal-3-pro speech model. MIPROv2 generates and evaluates
        candidate instructions to find the one that minimizes word error rate.
        The prompt provides domain context, expected vocabulary, or stylistic
        guidance to improve transcription accuracy.
        """

        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("audio_context -> transcription")
            self.predict.signature = self.predict.signature.with_instructions(starting_prompt)

        def forward(self, audio_context, sample_idx="0"):
            # Extract the instruction — this IS the AssemblyAI prompt
            prompt = self.predict.signature.instructions
            audio = samples[int(sample_idx)]["audio"]
            text = transcribe_assemblyai(audio, prompt, api_key)
            return dspy.Prediction(transcription=text)

    # Build trainset — each example is one audio sample
    trainset = [
        dspy.Example(
            audio_context="transcribe audio",
            sample_idx=str(i),
            reference=s["reference"],
        ).with_inputs("audio_context", "sample_idx")
        for i, s in enumerate(samples)
    ]

    # Run MIPROv2
    console.print(f"\n[bold cyan]Running MIPROv2 (auto={auto})...[/bold cyan]")
    optimizer = MIPROv2(
        metric=wer_metric,
        auto=auto,
        num_threads=num_threads,
        verbose=True,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
    )
    optimized = optimizer.compile(
        ASRModule(),
        trainset=trainset,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        requires_permission_to_run=False,
    )

    # Extract best prompt
    best_prompt = optimized.predict.signature.instructions

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = {
        "best_prompt": best_prompt,
        "starting_prompt": starting_prompt,
        "model": f"assemblyai/{SPEECH_MODEL}",
        "dataset": dataset,
        "split": split,
        "num_samples": len(samples),
        "optimizer": "MIPROv2",
        "auto": auto,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    state_file = output_path / "optimization_state.json"
    state_file.write_text(json.dumps(result, indent=2))

    console.print(f"\n[bold green]Best prompt: {best_prompt!r}[/bold green]")
    console.print(f"[dim]Saved to {state_file}[/dim]")


if __name__ == "__main__":
    app()
