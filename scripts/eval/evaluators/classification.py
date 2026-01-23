"""Classification evaluators for paralinguistic audio tasks (emotion, gender, age)."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import attrs
from rich.console import Console

from .asr import print_generation_config

console = Console()


@attrs.define
class ClassificationResult:
    """Result of a single classification evaluation."""

    prediction: str
    reference: str
    correct: bool
    time: float
    instruction: str
    task: str  # emotion, gender, age


class ClassificationEvaluator:
    """Evaluator for paralinguistic classification tasks using local models.

    Works with AudioLLMs-style datasets that have instruction/answer format,
    or raw datasets where instructions are generated dynamically.
    """

    # Keywords to extract class from model response
    EMOTION_KEYWORDS = [
        "angry",
        "anger",
        "happy",
        "happiness",
        "sad",
        "sadness",
        "fear",
        "fearful",
        "disgust",
        "disgusted",
        "surprise",
        "surprised",
        "neutral",
        "excited",
        "frustration",
        "frustrated",
    ]
    GENDER_KEYWORDS = ["female", "male", "woman", "man", "female_feminine", "male_masculine"]
    AGE_KEYWORDS = [
        "child",
        "teen",
        "teens",
        "teenager",
        "young",
        "adult",
        "middle",
        "elderly",
        "old",
        "senior",
        "twenties",
        "thirties",
        "forties",
        "fifties",
        "sixties",
        "seventies",
        "eighties",
        "nineties",
    ]

    # Default instructions for each task type (used when instruction_field is None)
    DEFAULT_INSTRUCTIONS = {
        "emotion": "What emotion is the speaker expressing?",
        "gender": "What is the gender of the speaker?",
        "age": "What is the approximate age group of the speaker?",
    }

    def __init__(
        self,
        model_path: str,
        audio_field: str = "audio",
        instruction_field: str | None = "instruction",
        answer_field: str = "answer",
        task: str = "emotion",  # emotion, gender, or age
        num_workers: int = 1,
    ):
        self.model_path = model_path
        self.audio_field = audio_field
        self.instruction_field = instruction_field  # None means generate dynamically
        self.answer_field = answer_field
        self.task = task
        self.num_workers = num_workers
        self.results: list[ClassificationResult] = []
        self._pipeline = None

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

    def _get_keywords(self) -> list[str]:
        """Get keywords for current task."""
        if self.task == "emotion":
            return self.EMOTION_KEYWORDS
        if self.task == "gender":
            return self.GENDER_KEYWORDS
        if self.task == "age":
            return self.AGE_KEYWORDS
        return []

    def extract_class(self, text: str) -> str:
        """Extract the predicted class from model output."""
        text_lower = text.lower()
        keywords = self._get_keywords()

        for keyword in keywords:
            if keyword in text_lower:
                return keyword

        # Return full text if no keyword found
        return text_lower.strip()

    def classify(self, audio: dict, instruction: str) -> tuple[str, float]:
        """Classify audio using the model."""
        start = time.time()
        try:
            result = self.pipeline(audio, generate_kwargs={"prompt": instruction})
        except Exception:
            # Model doesn't support prompts - just transcribe
            result = self.pipeline(audio)
        elapsed = time.time() - start

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text, elapsed

    def _process_sample(self, sample_data: tuple[int, dict]) -> tuple[int, ClassificationResult]:
        """Process a single classification sample."""
        idx, sample = sample_data

        try:
            prediction_raw, inference_time = self.classify(sample["audio"], sample["instruction"])
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            prediction_raw, inference_time = "", 0.0

        # Extract class from both prediction and reference
        pred_class = self.extract_class(prediction_raw)
        ref_class = self.extract_class(sample["reference"])

        # Check if prediction matches reference (keyword-based)
        correct = pred_class == ref_class

        return idx, ClassificationResult(
            prediction=prediction_raw,
            reference=sample["reference"],
            correct=correct,
            time=inference_time,
            instruction=sample["instruction"],
            task=self.task,
        )

    def evaluate(self, dataset, max_samples: int | None = None) -> list[ClassificationResult]:
        """Run evaluation loop on dataset."""
        self.results = []

        if self.num_workers > 1:
            samples = self._collect_samples(dataset, max_samples)
            self._evaluate_parallel(samples)
        else:
            self._evaluate_sequential(dataset, max_samples)

        return self.results

    def _get_instruction(self, sample: dict) -> str:
        """Get instruction from sample or generate default."""
        if self.instruction_field and self.instruction_field in sample:
            return sample[self.instruction_field]
        return self.DEFAULT_INSTRUCTIONS.get(self.task, "Describe what you hear.")

    def _get_reference(self, sample: dict) -> str:
        """Get reference answer, filtering out empty values."""
        ref = sample.get(self.answer_field, "")
        if ref is None or (isinstance(ref, str) and ref.strip() == ""):
            return ""
        return str(ref)

    def _collect_samples(self, dataset, max_samples: int | None) -> list[dict]:
        """Collect samples for parallel processing."""
        samples = []
        console.print(f"[dim]Collecting samples (target: {max_samples or 'all'})...[/dim]")

        for sample in dataset:
            reference = self._get_reference(sample)
            # Skip samples with empty reference
            if not reference:
                continue

            samples.append(
                {
                    "audio": sample[self.audio_field],
                    "instruction": self._get_instruction(sample),
                    "reference": reference,
                }
            )

            if len(samples) % 100 == 0:
                console.print(f"[dim]  Collected {len(samples)} samples...[/dim]")

            if max_samples and len(samples) >= max_samples:
                break

        console.print(f"[dim]Collected {len(samples)} samples, starting evaluation...[/dim]")
        return samples

    def _evaluate_sequential(self, dataset, max_samples: int | None) -> None:
        """Run sequential evaluation."""
        idx = 0
        for sample in dataset:
            reference = self._get_reference(sample)
            # Skip samples with empty reference
            if not reference:
                continue

            idx += 1
            sample_data = {
                "audio": sample[self.audio_field],
                "instruction": self._get_instruction(sample),
                "reference": reference,
            }

            _, result = self._process_sample((idx, sample_data))
            self.results.append(result)

            status = "correct" if result.correct else "wrong"
            print(f"Sample {idx}: {status} Time={result.time:.2f}s")
            print(f"  Pred: {result.prediction[:100]}")
            print(f"  Ref: {result.reference}")

            if idx % 100 == 0:
                self._print_checkpoint(idx)

            if max_samples and idx >= max_samples:
                break

    def _evaluate_parallel(self, samples: list[dict]) -> None:
        """Run parallel evaluation."""
        console.print(f"[bold]Running parallel evaluation with {self.num_workers} workers[/bold]")

        results_map: dict[int, ClassificationResult] = {}
        completed = 0
        total = len(samples)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._process_sample, (idx, sample)): idx
                for idx, sample in enumerate(samples, 1)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results_map[idx] = result
                completed += 1

                status = "correct" if result.correct else "wrong"
                print(f"[{completed}/{total}] Sample {idx}: {status} Time={result.time:.2f}s")
                print(f"  Pred: {result.prediction[:100]}")
                print(f"  Ref: {result.reference}")

                if completed % 100 == 0:
                    temp_results = list(results_map.values())
                    acc = sum(1 for r in temp_results if r.correct) / len(temp_results) * 100
                    console.print(f"\n[bold]CHECKPOINT @ {completed}[/bold]: Accuracy={acc:.2f}%\n")

        self.results = [results_map[i] for i in sorted(results_map.keys())]

    def _print_checkpoint(self, sample_count: int):
        """Print cumulative metrics checkpoint."""
        acc = sum(1 for r in self.results if r.correct) / len(self.results) * 100
        avg_time = sum(r.time for r in self.results) / len(self.results)
        console.print(
            f"\n[bold]CHECKPOINT @ {sample_count}[/bold]: Accuracy={acc:.2f}%, Avg Time={avg_time:.2f}s\n"
        )

    def compute_metrics(self) -> dict:
        """Compute final metrics."""
        if not self.results:
            return {"accuracy": 0.0, "avg_time": 0.0, "num_samples": 0}

        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)

        return {
            "accuracy": correct / total * 100,
            "correct": correct,
            "total": total,
            "avg_time": sum(r.time for r in self.results) / total,
            "num_samples": total,
            "task": self.task,
        }
