"""Classification evaluator implementations."""

import io
import time

from scripts.eval.audio import prepare_wav_bytes

from .base import ClassificationResult, console


def extract_label(text: str, valid_labels: list[str]) -> str | None:
    """Extract a classification label from model output.

    Handles various output formats:
    - Direct label: "Positive"
    - Sentence: "The sentiment is Positive."
    - Quoted: "The answer is 'Negative'"
    """
    text_lower = text.lower().strip()

    # Try exact match first (case-insensitive)
    for label in valid_labels:
        if label.lower() == text_lower:
            return label

    # Try to find label anywhere in text
    for label in valid_labels:
        if label.lower() in text_lower:
            return label

    return None


class BaseClassificationEvaluator:
    """Base class for classification evaluators."""

    def __init__(
        self,
        audio_field: str = "audio",
        label_field: str = "label",
        instruction_field: str | None = None,
        valid_labels: list[str] | None = None,
        default_instruction: str | None = None,
        num_workers: int = 1,
    ):
        self.audio_field = audio_field
        self.label_field = label_field
        self.instruction_field = instruction_field
        self.valid_labels = valid_labels or []
        self.default_instruction = default_instruction
        self.num_workers = num_workers
        self.results: list[ClassificationResult] = []

    def classify(self, audio, instruction: str) -> tuple[str, float]:
        """Classify audio and return (raw_output, inference_time). Override in subclass."""
        raise NotImplementedError

    def evaluate(self, dataset, max_samples: int | None = None) -> list[ClassificationResult]:
        """Run classification evaluation loop on dataset."""
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
        target = max_samples or "all"
        console.print(f"[dim]Collecting samples (target: {target})...[/dim]")

        for sample in dataset:
            # Get instruction from sample or use default
            if self.instruction_field and self.instruction_field in sample:
                instruction = sample[self.instruction_field]
            else:
                instruction = self.default_instruction or ""

            samples.append(
                {
                    "audio": sample[self.audio_field],
                    "reference": sample[self.label_field],
                    "instruction": instruction,
                }
            )

            if len(samples) % 50 == 0:
                console.print(f"[dim]  Collected {len(samples)} samples...[/dim]")

            if max_samples and len(samples) >= max_samples:
                break

        console.print(f"[dim]Collected {len(samples)} samples, starting evaluation...[/dim]")
        return samples

    def _process_sample(self, sample_data: tuple[int, dict]) -> tuple[int, ClassificationResult]:
        """Process a single sample. Returns (index, result) for ordering."""
        idx, sample = sample_data

        try:
            raw_output, inference_time = self.classify(sample["audio"], sample["instruction"])
        except Exception as e:
            console.print(f"[red]Error on sample {idx}: {e}[/red]")
            return idx, ClassificationResult(
                prediction="",
                reference=sample["reference"],
                correct=False,
                time=0.0,
                raw_output=f"Error: {e}",
            )

        prediction = extract_label(raw_output, self.valid_labels)
        if prediction is None:
            prediction = raw_output.strip()[:50]

        correct = prediction.lower() == sample["reference"].lower()

        return idx, ClassificationResult(
            prediction=prediction,
            reference=sample["reference"],
            correct=correct,
            time=inference_time,
            raw_output=raw_output,
        )

    def _evaluate_parallel(self, samples: list[dict]) -> None:
        """Run parallel evaluation using thread pool."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

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

                status = "[green]CORRECT[/green]" if result.correct else "[red]WRONG[/red]"
                console.print(
                    f"[{completed}/{total}] Sample {idx}: {status} "
                    f"(pred={result.prediction}, ref={result.reference}) "
                    f"Time={result.time:.2f}s"
                )

                if completed % 50 == 0:
                    temp_results = list(results_map.values())
                    num_correct = sum(1 for r in temp_results if r.correct)
                    accuracy = num_correct / len(temp_results) * 100
                    console.print(
                        f"\n[bold]CHECKPOINT @ {completed}[/bold]: Accuracy={accuracy:.1f}%\n"
                    )

        self.results = [results_map[i] for i in sorted(results_map.keys())]

    def _evaluate_sequential(self, dataset, max_samples: int | None) -> None:
        """Run sequential evaluation."""
        processed = 0

        for sample in dataset:
            processed += 1
            if max_samples and processed > max_samples:
                break

            reference = sample[self.label_field]

            if self.instruction_field and self.instruction_field in sample:
                instruction = sample[self.instruction_field]
            else:
                instruction = self.default_instruction or ""

            try:
                raw_output, inference_time = self.classify(sample[self.audio_field], instruction)
            except Exception as e:
                console.print(f"[red]Error on sample {processed}: {e}[/red]")
                continue

            prediction = extract_label(raw_output, self.valid_labels)
            if prediction is None:
                prediction = raw_output.strip()[:50]

            correct = prediction.lower() == reference.lower()

            result = ClassificationResult(
                prediction=prediction,
                reference=reference,
                correct=correct,
                time=inference_time,
                raw_output=raw_output,
            )
            self.results.append(result)

            status = "[green]CORRECT[/green]" if correct else "[red]WRONG[/red]"
            console.print(
                f"Sample {processed}: {status} "
                f"(pred={prediction}, ref={reference}) "
                f"Time={inference_time:.2f}s"
            )

            if processed % 50 == 0:
                self._print_checkpoint(processed)

    def _print_checkpoint(self, sample_count: int):
        """Print cumulative metrics checkpoint."""
        metrics = self.compute_metrics()
        console.print(f"\n[bold]{'=' * 60}[/bold]")
        console.print(
            f"[bold]CHECKPOINT @ {sample_count}:[/bold] Accuracy={metrics['accuracy']:.1f}%"
        )
        console.print(f"[bold]{'=' * 60}[/bold]\n")

    def compute_metrics(self) -> dict:
        """Compute final metrics."""
        if not self.results:
            return {
                "accuracy": 0.0,
                "num_correct": 0,
                "num_samples": 0,
                "avg_time": 0.0,
            }

        num_correct = sum(1 for r in self.results if r.correct)
        accuracy = num_correct / len(self.results) * 100

        return {
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_samples": len(self.results),
            "avg_time": sum(r.time for r in self.results) / len(self.results),
        }


class LocalClassificationEvaluator(BaseClassificationEvaluator):
    """Classification evaluator for local models."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        from transformers import pipeline

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            trust_remote_code=True,
        )

    def classify(self, audio, instruction: str) -> tuple[str, float]:
        start = time.time()
        result = self.pipe(audio, user_prompt=instruction)
        elapsed = time.time() - start
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text, elapsed


class AssemblyAIClassificationEvaluator(BaseClassificationEvaluator):
    """Classification evaluator using AssemblyAI slam_1 with prompt parameter."""

    def __init__(
        self,
        api_key: str,
        model: str = "slam_1",
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        import assemblyai as aai

        aai.settings.api_key = api_key
        if base_url:
            aai.settings.base_url = base_url

        self.model = model
        self._aai = aai

    def classify(self, audio, instruction: str) -> tuple[str, float]:
        wav_bytes = prepare_wav_bytes(audio)

        # Create config with prompt parameter for slam_1
        config = self._aai.TranscriptionConfig(
            speech_model=getattr(self._aai.types.SpeechModel, self.model),
            prompt=instruction,
        )
        transcriber = self._aai.Transcriber(config=config)

        start = time.time()
        transcript = transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start

        return transcript.text or "", elapsed
