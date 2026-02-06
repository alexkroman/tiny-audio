"""AIR-Bench evaluator for audio understanding benchmarks.

Implements the AIR-Bench Foundation benchmark evaluation following the methodology
from https://github.com/OFA-Sys/AIR-Bench
"""

import json
import time
from pathlib import Path

import attrs
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.table import Table

from tiny_audio.asr_pipeline import strip_thinking

from .asr import print_generation_config

console = Console()

# Speech-related tasks in AIR-Bench Foundation
SPEECH_TASKS = {
    "Speech_Grounding",
    "Spoken_Language_Identification",
    "Speaker_Gender_Recognition",
    "Speaker_Emotion_Recontion",  # Note: typo in original dataset
    "Speaker_Age_Prediction",
    "Speech_Entity_Reconition",  # Note: typo in original dataset
    "Speaker_Intent_Classification",
    "Speaker_Number_Verification",
    "Synthesized_Voice_Detection",
}

# Sound-related tasks
SOUND_TASKS = {
    "Audio_Grounding",
    "vocal_sound_classification",
    "Acoustic_Scene_Classification",
    "Sound_AQA",
}

# Music-related tasks
MUSIC_TASKS = {
    "Music_Instruments_Classfication",  # Note: typo in original dataset
    "Music_Genre_Recognition",
    "Music_Midi_Pitch_Analysis",
    "Music_Midi_Velocity_Analysis",
    "Music_AQA",
    "Music_Mood_Recognition",
}

ALL_TASKS = SPEECH_TASKS | SOUND_TASKS | MUSIC_TASKS

# AIR-Bench prompt template (from Inference_Foundation.py)
AIRBENCH_PROMPT_TEMPLATE = """Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D.
{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
Respond with exactly one letter."""


@attrs.define
class AIRBenchResult:
    """Result of a single AIR-Bench evaluation."""

    prediction: str
    extracted_answer: str  # A, B, C, or D
    reference: str  # The ground truth letter (A, B, C, or D)
    correct: bool
    time: float
    question: str
    task_name: str
    dataset_name: str
    uniq_id: int


class AIRBenchEvaluator:
    """Evaluator for AIR-Bench Foundation benchmark using local models.

    Follows the evaluation methodology from https://github.com/OFA-Sys/AIR-Bench
    """

    def __init__(
        self,
        model_path: str,
        task_name: str | None = None,  # Filter to specific task, None = all speech tasks
        num_workers: int = 1,
    ):
        self.model_path = model_path
        self.task_name = task_name
        self.num_workers = num_workers
        self.results: list[AIRBenchResult] = []
        self._pipeline = None
        self._metadata: list[dict] | None = None
        self._audio_cache_dir: Path | None = None

    @property
    def pipeline(self):
        """Lazy-load the ASR pipeline."""
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_path,
                trust_remote_code=True,
            )
            print_generation_config(self._pipeline.model, self.model_path)
        return self._pipeline

    def _load_metadata(self) -> list[dict]:
        """Load AIR-Bench Foundation metadata from HuggingFace Hub."""
        if self._metadata is not None:
            return self._metadata

        console.print("[dim]Downloading AIR-Bench Foundation metadata...[/dim]")
        meta_file = hf_hub_download(
            repo_id="qyang1021/AIR-Bench-Dataset",
            filename="Foundation/Foundation_meta.json",
            repo_type="dataset",
        )

        with Path(meta_file).open() as f:
            self._metadata = json.load(f)

        # Store the cache directory for audio files
        self._audio_cache_dir = Path(meta_file).parent.parent

        console.print(f"[dim]Loaded {len(self._metadata)} items from AIR-Bench Foundation[/dim]")
        return self._metadata

    def _get_audio_path(self, item: dict) -> Path:
        """Get the local path for an audio file, downloading if needed."""
        task_name = item["task_name"]
        dataset_name = item["dataset_name"]
        audio_filename = item["path"]

        # Handle Audio_Grounding special case (flac extension)
        if task_name == "Audio_Grounding":
            audio_filename = audio_filename.rsplit(".", 1)[0] + ".flac"

        # Construct the HF Hub path
        hf_path = f"Foundation/{task_name}_{dataset_name}/{audio_filename}"

        # Download the audio file
        try:
            local_path = hf_hub_download(
                repo_id="qyang1021/AIR-Bench-Dataset",
                filename=hf_path,
                repo_type="dataset",
            )
            return Path(local_path)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not download {hf_path}: {e}[/yellow]")
            return None

    def _construct_prompt(self, item: dict) -> str:
        """Construct the AIR-Bench prompt following the official format."""
        return AIRBENCH_PROMPT_TEMPLATE.format(
            question=item["question"],
            choice_a=item["choice_a"],
            choice_b=item["choice_b"],
            choice_c=item.get("choice_c", "N/A"),
            choice_d=item.get("choice_d", "N/A"),
        )

    def _get_ground_truth_letter(self, item: dict) -> str:
        """Convert answer_gt to letter (A, B, C, or D)."""
        answer_gt = item["answer_gt"]

        if answer_gt == item["choice_a"]:
            return "A"
        if answer_gt == item["choice_b"]:
            return "B"
        if answer_gt == item.get("choice_c"):
            return "C"
        if answer_gt == item.get("choice_d"):
            return "D"

        # Fallback: check string representation for list answers
        answer_str = str(answer_gt)
        if answer_str == str(item["choice_a"]):
            return "A"
        if answer_str == str(item["choice_b"]):
            return "B"
        if answer_str == str(item.get("choice_c")):
            return "C"
        if answer_str == str(item.get("choice_d")):
            return "D"

        console.print(
            f"[yellow]Warning: Could not match answer_gt '{answer_gt}' to choices[/yellow]"
        )
        return "?"

    def _extract_answer(self, response: str) -> str:
        """Extract A, B, C, or D from model response."""
        import re

        if not response:
            return "?"

        # Check if response is ONLY a single letter (direct answer)
        if response.strip() in ("A", "B", "C", "D"):
            return response.strip()

        # Look for standalone letter at end: "...the answer is A" or just "A" on its own line
        match = re.search(r"\b([ABCD])\s*$", response)
        if match:
            return match.group(1)

        # Look for quoted letter at end: respond with "C" or 'C'
        match = re.search(r'["\']([ABCD])["\']\s*\.?\s*$', response)
        if match:
            return match.group(1)

        # Look for parenthesized letter: (A), (B), (C), (D) - common format
        matches = re.findall(r"\(([ABCD])\)", response)
        if matches:
            return matches[0]  # Return first match (usually the answer stated early)

        # Look for "A." / "B." / "C." / "D." pattern at end of response
        match = re.search(r"\b([ABCD])\.\s*\w+\s*$", response)
        if match:
            return match.group(1)

        # Look for "answer/option is X" or "answer: X" pattern
        match = re.search(r"(?:answer|option)\s*(?:is|:)\s*([ABCD])\b", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Look for "X. <word>" pattern (e.g., "C. Train") - find LAST occurrence
        matches = re.findall(r"\b([ABCD])\.\s+\w+", response)
        if matches:
            return matches[-1]  # Return last match

        # Last resort: find the LAST letter mentioned (reasoning comes first, answer last)
        for letter in ("D", "C", "B", "A"):  # Reverse order to prefer later mentions
            # Find all positions of this letter
            positions = [i for i, c in enumerate(response.upper()) if c == letter]
            # Check if this letter appears in the second half of the response
            if positions and positions[-1] > len(response) // 2:
                return letter

        # Absolute fallback: any letter
        for letter in ("A", "B", "C", "D"):
            if letter in response.upper():
                return letter

        return "?"

    def _process_sample(self, item: dict) -> AIRBenchResult | None:
        """Process a single AIR-Bench sample."""
        audio_path = self._get_audio_path(item)
        if audio_path is None:
            return None

        prompt = self._construct_prompt(item)
        gt_letter = self._get_ground_truth_letter(item)

        start = time.time()
        try:
            result = self.pipeline(
                str(audio_path),
                user_prompt=prompt,
            )
            prediction = result.get("text", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            console.print(f"[red]Error processing sample {item['uniq_id']}: {e}[/red]")
            prediction = ""
        elapsed = time.time() - start

        # Strip thinking tags for display and extraction
        prediction = strip_thinking(prediction)
        extracted = self._extract_answer(prediction)
        correct = extracted == gt_letter

        return AIRBenchResult(
            prediction=prediction,
            extracted_answer=extracted,
            reference=gt_letter,
            correct=correct,
            time=elapsed,
            question=item["question"],
            task_name=item["task_name"],
            dataset_name=item["dataset_name"],
            uniq_id=item["uniq_id"],
        )

    def evaluate(self, max_samples: int | None = None) -> list[AIRBenchResult]:
        """Run evaluation on AIR-Bench Foundation.

        Args:
            max_samples: Maximum number of samples to evaluate.
                - If a specific task is selected: limits total samples
                - If running all tasks: limits samples PER TASK for balanced evaluation

        Returns:
            List of AIRBenchResult objects.
        """
        self.results = []
        metadata = self._load_metadata()

        # Filter by task_name if specified
        if self.task_name:
            # Single task: max_samples is total limit
            items = [item for item in metadata if item["task_name"] == self.task_name]
            console.print(
                f"[bold]Filtering to task: {self.task_name} ({len(items)} samples)[/bold]"
            )
            if max_samples:
                items = items[:max_samples]
        else:
            # Multiple tasks: max_samples is PER TASK for balanced evaluation
            # Default to speech tasks only (use --task to run specific task)
            task_set = SPEECH_TASKS
            items = []

            # Group by task and take max_samples from each
            task_items: dict[str, list[dict]] = {}
            for item in metadata:
                if item["task_name"] in task_set:
                    task_name = item["task_name"]
                    if task_name not in task_items:
                        task_items[task_name] = []
                    task_items[task_name].append(item)

            # Take max_samples from each task
            for task_name in sorted(task_items.keys()):
                task_list = task_items[task_name]
                if max_samples:
                    task_list = task_list[:max_samples]
                items.extend(task_list)
                console.print(f"[dim]  {task_name}: {len(task_list)} samples[/dim]")

            console.print(
                f"[bold]Evaluating {len(task_items)} tasks "
                f"({len(items)} total samples, {max_samples or 'all'} per task)[/bold]"
            )

        console.print(f"[bold]Starting evaluation on {len(items)} samples...[/bold]")

        for idx, item in enumerate(items, 1):
            result = self._process_sample(item)
            if result is None:
                continue

            self.results.append(result)

            status = "✓" if result.correct else "✗"
            console.print(
                f"[{idx}/{len(items)}] {status} {result.task_name} | "
                f"Pred: {result.extracted_answer} | GT: {result.reference} | "
                f"Time: {result.time:.2f}s"
            )
            console.print(f"  [dim]Raw output: {result.prediction or '(empty)'}[/dim]")

            if idx % 100 == 0:
                self._print_checkpoint(idx)

        return self.results

    def _print_checkpoint(self, sample_count: int):
        """Print cumulative metrics checkpoint."""
        if not self.results:
            return

        acc = sum(1 for r in self.results if r.correct) / len(self.results) * 100
        avg_time = sum(r.time for r in self.results) / len(self.results)
        console.print(
            f"\n[bold]CHECKPOINT @ {sample_count}[/bold]: Accuracy={acc:.2f}%, Avg Time={avg_time:.2f}s\n"
        )

    def compute_metrics(self) -> dict:
        """Compute final metrics with per-task breakdown.

        Returns metrics following AIR-Bench methodology:
        - Overall accuracy
        - Per-task accuracy (task_name + dataset_name)
        - Per-category accuracy (speech, sound, music)
        """
        if not self.results:
            return {"accuracy": 0.0, "avg_time": 0.0, "num_samples": 0}

        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)

        # Per-task accuracy (task_name_dataset_name, following AIR-Bench)
        task_results: dict[str, list[AIRBenchResult]] = {}
        for r in self.results:
            task_id = f"{r.task_name}_{r.dataset_name}"
            if task_id not in task_results:
                task_results[task_id] = []
            task_results[task_id].append(r)

        task_accuracy = {}
        for task_id, results in sorted(task_results.items()):
            task_correct = sum(1 for r in results if r.correct)
            task_accuracy[task_id] = {
                "accuracy": task_correct / len(results) * 100,
                "correct": task_correct,
                "total": len(results),
            }

        # Per-category accuracy (speech, sound, music)
        category_results: dict[str, list[AIRBenchResult]] = {"speech": [], "sound": [], "music": []}
        for r in self.results:
            if r.task_name in SPEECH_TASKS:
                category_results["speech"].append(r)
            elif r.task_name in SOUND_TASKS:
                category_results["sound"].append(r)
            elif r.task_name in MUSIC_TASKS:
                category_results["music"].append(r)

        category_accuracy = {}
        for category, results in category_results.items():
            if results:
                cat_correct = sum(1 for r in results if r.correct)
                category_accuracy[category] = cat_correct / len(results) * 100

        return {
            "accuracy": correct / total * 100,
            "correct": correct,
            "total": total,
            "avg_time": sum(r.time for r in self.results) / total,
            "num_samples": total,
            "task_accuracy": task_accuracy,
            "category_accuracy": category_accuracy,
        }


def print_airbench_metrics(metrics: dict):
    """Print AIR-Bench metrics using rich tables."""
    # Overall metrics
    overall_table = Table(title="AIR-Bench Foundation Results")
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", style="green")

    overall_table.add_row("Overall Accuracy", f"{metrics['accuracy']:.2f}%")
    overall_table.add_row("Correct / Total", f"{metrics['correct']} / {metrics['total']}")
    overall_table.add_row("Avg Time", f"{metrics['avg_time']:.2f}s")

    console.print(overall_table)

    # Category breakdown
    if metrics.get("category_accuracy"):
        cat_table = Table(title="Per-Category Accuracy")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Accuracy", style="green")

        for category, acc in sorted(metrics["category_accuracy"].items()):
            cat_table.add_row(category.title(), f"{acc:.2f}%")

        console.print(cat_table)

    # Per-task breakdown
    if metrics.get("task_accuracy"):
        task_table = Table(title="Per-Task Accuracy")
        task_table.add_column("Task", style="cyan")
        task_table.add_column("Accuracy", style="green")
        task_table.add_column("Correct/Total", style="dim")

        for task_id, task_metrics in sorted(metrics["task_accuracy"].items()):
            task_table.add_row(
                task_id,
                f"{task_metrics['accuracy']:.2f}%",
                f"{task_metrics['correct']}/{task_metrics['total']}",
            )

        console.print(task_table)
