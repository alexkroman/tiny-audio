"""MCQ (Multiple Choice Question) evaluators for audio understanding benchmarks."""

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import attrs
from rich.console import Console

from scripts.eval.audio import prepare_wav_bytes

console = Console()


@attrs.define
class MCQResult:
    """Result of a single MCQ evaluation."""

    prediction: str
    matched_choice: str
    reference: str
    correct: bool
    time: float
    question: str
    choices: list[str]
    category: str


class MMAUEvaluator:
    """Evaluator for MMAU benchmark using local models."""

    def __init__(
        self,
        model_path: str,
        audio_field: str = "context",
        question_field: str = "instruction",
        answer_field: str = "answer",
        choices_field: str = "choices",
        category_field: str = "other_attributes",
        user_prompt: str | None = None,
        num_workers: int = 1,
    ):
        self.model_path = model_path
        self.audio_field = audio_field
        self.question_field = question_field
        self.answer_field = answer_field
        self.choices_field = choices_field
        self.category_field = category_field
        self.user_prompt = user_prompt
        self.num_workers = num_workers
        self.results: list[MCQResult] = []
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
        return self._pipeline

    def _extract_category(self, other_attributes) -> str:
        """Extract category from other_attributes JSON string."""
        try:
            category = None
            if isinstance(other_attributes, str):
                attrs_dict = json.loads(other_attributes)
                category = attrs_dict.get("category", "unknown")
            elif isinstance(other_attributes, dict):
                category = other_attributes.get("category", "unknown")
            else:
                category = other_attributes

            # Handle list values like ['Reasoning']
            if isinstance(category, list) and len(category) > 0:
                category = category[0]

            return str(category) if category else "unknown"
        except (json.JSONDecodeError, TypeError, ValueError):
            return "unknown"

    def answer_question(self, audio: dict, question: str, choices: list[str]) -> tuple[str, float]:
        """Answer a question about audio using the model."""
        # Format prompt with question and choices
        choices_str = "\n".join(choices)
        prompt = f"{question}\n{choices_str}\n\nRespond with only the letter of the correct answer (A, B, C, or D)."

        if self.user_prompt:
            prompt = f"{self.user_prompt}\n{prompt}"

        start = time.time()
        # Try with prompt first, fall back to plain transcription if not supported
        try:
            result = self.pipeline(audio, generate_kwargs={"prompt": prompt})
        except Exception:
            # Model doesn't support prompts - just transcribe
            result = self.pipeline(audio)
        elapsed = time.time() - start

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text, elapsed

    def match_to_choice(self, prediction: str, choices: list[str]) -> str:
        """Match model prediction to one of the available choices."""
        prediction_lower = prediction.lower().strip()

        # Direct match
        for choice in choices:
            if choice.lower() == prediction_lower:
                return choice

        # Check if prediction contains choice letter like "(A)" or "A"
        for choice in choices:
            # Extract letter from choice like "(A) Man" -> "A"
            if choice.startswith("(") and ")" in choice:
                letter = choice[1 : choice.index(")")]
                if f"({letter})" in prediction or prediction_lower.startswith(letter.lower()):
                    return choice

        # Substring match
        for choice in choices:
            if choice.lower() in prediction_lower:
                return choice

        return prediction

    def _process_sample(self, sample_data: tuple[int, dict]) -> tuple[int, MCQResult]:
        """Process a single MCQ sample."""
        idx, sample = sample_data

        try:
            prediction, inference_time = self.answer_question(
                sample["audio"], sample["question"], sample["choices"]
            )
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            prediction, inference_time = "", 0.0

        matched = self.match_to_choice(prediction, sample["choices"])
        correct = matched.lower().strip() == sample["reference"].lower().strip()

        return idx, MCQResult(
            prediction=prediction,
            matched_choice=matched,
            reference=sample["reference"],
            correct=correct,
            time=inference_time,
            question=sample["question"],
            choices=sample["choices"],
            category=sample["category"],
        )

    def evaluate(self, dataset, max_samples: int | None = None) -> list[MCQResult]:
        """Run evaluation loop on dataset."""
        self.results = []

        if self.num_workers > 1:
            samples = self._collect_samples(dataset, max_samples)
            self._evaluate_parallel(samples)
        else:
            self._evaluate_sequential(dataset, max_samples)

        return self.results

    def _collect_samples(self, dataset, max_samples: int | None) -> list[dict]:
        """Collect samples for parallel processing."""
        samples = []
        console.print(f"[dim]Collecting samples (target: {max_samples or 'all'})...[/dim]")

        for sample in dataset:
            samples.append(
                {
                    "audio": sample[self.audio_field],
                    "question": sample[self.question_field],
                    "reference": sample[self.answer_field],
                    "choices": sample[self.choices_field],
                    "category": self._extract_category(sample.get(self.category_field, "")),
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
        for idx, sample in enumerate(dataset, 1):
            sample_data = {
                "audio": sample[self.audio_field],
                "question": sample[self.question_field],
                "reference": sample[self.answer_field],
                "choices": sample[self.choices_field],
                "category": self._extract_category(sample.get(self.category_field, "")),
            }

            _, result = self._process_sample((idx, sample_data))
            self.results.append(result)

            status = "✓" if result.correct else "✗"
            print(f"Sample {idx}: {status} Time={result.time:.2f}s")
            print(f"  Q: {result.question}")
            print(f"  Pred: {result.prediction[:100]}")
            print(f"  Match: {result.matched_choice} | Ref: {result.reference}")

            if idx % 100 == 0:
                self._print_checkpoint(idx)

            if max_samples and idx >= max_samples:
                break

    def _evaluate_parallel(self, samples: list[dict]) -> None:
        """Run parallel evaluation."""
        console.print(f"[bold]Running parallel evaluation with {self.num_workers} workers[/bold]")

        results_map: dict[int, MCQResult] = {}
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

                status = "✓" if result.correct else "✗"
                print(f"[{completed}/{total}] Sample {idx}: {status} Time={result.time:.2f}s")
                print(f"  Q: {result.question}")
                print(f"  Pred: {result.prediction[:100]}")
                print(f"  Match: {result.matched_choice} | Ref: {result.reference}")

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

        # Per-category accuracy
        categories: dict[str, list[MCQResult]] = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)

        category_acc = {
            cat: sum(1 for r in results if r.correct) / len(results) * 100
            for cat, results in categories.items()
        }

        return {
            "accuracy": correct / total * 100,
            "correct": correct,
            "total": total,
            "avg_time": sum(r.time for r in self.results) / total,
            "num_samples": total,
            "category_accuracy": category_acc,
        }


class AssemblyAIMMAUEvaluator(MMAUEvaluator):
    """Evaluator for MMAU benchmark using AssemblyAI SLAM model with prompt support."""

    def __init__(
        self,
        api_key: str,
        model: str = "slam_1",
        audio_field: str = "context",
        question_field: str = "instruction",
        answer_field: str = "answer",
        choices_field: str = "choices",
        category_field: str = "other_attributes",
        num_workers: int = 1,
    ):
        # Don't call parent __init__ fully - we override the pipeline
        self.audio_field = audio_field
        self.question_field = question_field
        self.answer_field = answer_field
        self.choices_field = choices_field
        self.category_field = category_field
        self.user_prompt = None
        self.num_workers = num_workers
        self.results: list[MCQResult] = []

        self.api_key = api_key
        self.model = model

    def answer_question(self, audio: dict, question: str, choices: list[str]) -> tuple[str, float]:
        """Answer a question about audio using AssemblyAI with prompt."""
        import assemblyai as aai

        aai.settings.api_key = self.api_key

        # Format prompt with question and choices
        choices_str = "\n".join(choices)
        prompt = f"{question}\n{choices_str}\n\nRespond with only the letter of the correct answer (A, B, C, or D)."

        # Create config with prompt (undocumented but supported)
        config = aai.TranscriptionConfig(
            speech_model=getattr(aai.types.SpeechModel, self.model),
            prompt=prompt,
        )
        transcriber = aai.Transcriber(config=config)

        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()
        transcript = transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start

        time.sleep(0.5)  # Rate limiting
        return transcript.text or "", elapsed
