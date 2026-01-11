"""ASR evaluator implementations."""

import io
import tempfile
import time
from pathlib import Path

import soundfile as sf
import torch

from scripts.eval.audio import prepare_wav_bytes

from .base import Evaluator, console, setup_assemblyai


def print_generation_config(model, model_path: str):
    """Print generation config in a visible format."""
    gen_config = model.generation_config
    console.print(
        "\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]"
    )
    console.print(f"[bold]Model:[/bold] {model_path}")
    console.print("[bold cyan]Generation Config:[/bold cyan]")
    console.print(f"  max_new_tokens:      {gen_config.max_new_tokens}")
    console.print(f"  min_new_tokens:      {gen_config.min_new_tokens}")
    console.print(f"  num_beams:           {gen_config.num_beams}")
    console.print(f"  do_sample:           {gen_config.do_sample}")
    console.print(f"  repetition_penalty:  {gen_config.repetition_penalty}")
    console.print(f"  length_penalty:      {gen_config.length_penalty}")
    console.print(f"  no_repeat_ngram_size: {gen_config.no_repeat_ngram_size}")
    console.print(
        "[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n"
    )


class LocalEvaluator(Evaluator):
    """Evaluator for local models."""

    def __init__(self, model_path: str, user_prompt: str | None = None, **kwargs):
        super().__init__(**kwargs)
        from transformers import pipeline

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            trust_remote_code=True,
        )
        self.user_prompt = user_prompt

        print_generation_config(self.pipe.model, model_path)

    def transcribe(self, audio) -> tuple[str, float]:
        start = time.time()
        result = self.pipe(audio, user_prompt=self.user_prompt)
        elapsed = time.time() - start
        return result.get("text", "") if isinstance(result, dict) else str(result), elapsed


class LocalStreamingEvaluator(Evaluator):
    """Evaluator for local models with streaming metrics (TTFB, processing time)."""

    def __init__(self, model_path: str, user_prompt: str | None = None, **kwargs):
        super().__init__(**kwargs)
        from transformers import pipeline

        # Determine best device and dtype
        if torch.cuda.is_available():
            device = 0
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = -1
            dtype = torch.float32

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            trust_remote_code=True,
            device=device,
            torch_dtype=dtype,
        )
        self.model = self.pipe.model
        self.model.eval()
        self.processor = self.model.get_processor()
        self.user_prompt = user_prompt
        print(f"Using device: {device}, dtype: {dtype}")

        # Track timing stats
        self.ttfb_times: list[float] = []
        self.processing_times: list[float] = []

        # Print generation config
        print_generation_config(self.model, model_path)

    def transcribe(self, audio) -> tuple[str, float]:
        import threading

        from transformers import TextIteratorStreamer

        # Extract audio array
        if isinstance(audio, dict) and "array" in audio:
            audio_array = audio["array"]
            sample_rate = audio.get("sampling_rate", 16000)
        elif isinstance(audio, dict) and "raw" in audio:
            audio_array = audio["raw"]
            sample_rate = audio.get("sampling_rate", 16000)
        else:
            wav_bytes = prepare_wav_bytes(audio)
            audio_array, sample_rate = sf.read(io.BytesIO(wav_bytes))

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa

            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

        # Process audio (ASRProcessor handles sampling_rate internally)
        inputs = self.processor(
            audio_array,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(
            device=self.model.device, dtype=self.model.dtype
        )
        audio_attention_mask = inputs["audio_attention_mask"].to(self.model.device)

        # Set up streamer to capture first token time
        streamer = TextIteratorStreamer(
            self.model.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        first_token_time = [None]
        generation_start = [None]

        def generate():
            generation_start[0] = time.time()
            self.model.generate(
                input_features=input_features,
                audio_attention_mask=audio_attention_mask,
                streamer=streamer,
            )

        # Start generation in background thread
        thread = threading.Thread(target=generate)
        thread.start()

        # Collect tokens and measure TTFB
        tokens = []
        for text in streamer:
            if first_token_time[0] is None and text:
                first_token_time[0] = time.time()
            tokens.append(text)

        thread.join()
        processing_end = time.time()

        # Calculate timing metrics
        processing_time = processing_end - generation_start[0] if generation_start[0] else 0
        ttfb = (
            (first_token_time[0] - generation_start[0])
            if first_token_time[0] and generation_start[0]
            else None
        )

        # Store for aggregation
        self.processing_times.append(processing_time)
        if ttfb is not None:
            self.ttfb_times.append(ttfb)

        # Print timing info
        ttfb_str = f"{ttfb * 1000:.0f}ms" if ttfb else "N/A"
        print(f"  [Streaming] TTFB: {ttfb_str}, Processing: {processing_time * 1000:.0f}ms")

        full_text = "".join(tokens).strip()
        # Apply pipeline post-processing (repetition truncation, etc.)
        full_text = self.pipe._post_process_prediction(full_text)
        return full_text, processing_time

    def compute_metrics(self) -> dict:
        """Compute final metrics including streaming-specific timing."""
        metrics = super().compute_metrics()
        if self.ttfb_times:
            metrics["avg_ttfb"] = sum(self.ttfb_times) / len(self.ttfb_times)
            metrics["min_ttfb"] = min(self.ttfb_times)
            metrics["max_ttfb"] = max(self.ttfb_times)
        if self.processing_times:
            metrics["avg_processing"] = sum(self.processing_times) / len(self.processing_times)
        return metrics


class EndpointEvaluator(Evaluator):
    """Evaluator for HuggingFace Inference Endpoints."""

    def __init__(self, endpoint_url: str, **kwargs):
        super().__init__(**kwargs)
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(base_url=endpoint_url)
        self.temp_dir = tempfile.mkdtemp()

    def transcribe(self, audio) -> tuple[str, float]:
        wav_bytes = prepare_wav_bytes(audio)
        temp_path = Path(self.temp_dir) / f"temp_{time.time_ns()}.wav"
        temp_path.write_bytes(wav_bytes)

        try:
            start = time.time()
            result = self.client.automatic_speech_recognition(str(temp_path))
            elapsed = time.time() - start

            if isinstance(result, dict):
                text = result.get("text", result.get("transcription", ""))
            elif hasattr(result, "text"):
                text = result.text
            else:
                text = str(result)
            return text, elapsed
        finally:
            temp_path.unlink(missing_ok=True)

    def __del__(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


class AssemblyAIEvaluator(Evaluator):
    """Evaluator for AssemblyAI API."""

    def __init__(self, api_key: str, model: str = "slam_1", base_url: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.transcriber = setup_assemblyai(api_key, model, base_url=base_url)

    def transcribe(self, audio) -> tuple[str, float]:
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()
        transcript = self.transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start
        return transcript.text or "", elapsed


class AssemblyAIStreamingEvaluator(Evaluator):
    """Evaluator for AssemblyAI Streaming API (Universal-Streaming model)."""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        # Track timing stats across samples
        self.ttfb_times: list[float] = []
        self.processing_times: list[float] = []

    def transcribe(self, audio) -> tuple[str, float]:
        import threading

        import numpy as np
        import soundfile as sf
        from assemblyai.streaming.v3 import (
            StreamingClient,
            StreamingClientOptions,
            StreamingEvents,
            StreamingParameters,
        )

        # Convert audio to raw PCM bytes (16kHz, 16-bit mono)
        if isinstance(audio, dict) and "array" in audio:
            audio_array = audio["array"]
            sample_rate = audio.get("sampling_rate", 16000)
        else:
            wav_bytes = prepare_wav_bytes(audio)
            audio_array, sample_rate = sf.read(io.BytesIO(wav_bytes))

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa

            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

        # Convert to 16-bit PCM bytes
        if isinstance(audio_array, np.ndarray):
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / np.abs(audio_array).max()
            pcm_data = (audio_array * 32767).astype(np.int16).tobytes()
        else:
            pcm_data = audio_array

        # State for this transcription
        transcripts = {}
        error_occurred = [None]
        session_done = threading.Event()
        first_transcript_time = [None]
        stream_start_time = [None]
        final_transcript_time = [None]

        def on_turn(client, event):
            if event.transcript:
                # Record time to first transcript (any transcript, not just final)
                if first_transcript_time[0] is None and stream_start_time[0] is not None:
                    first_transcript_time[0] = time.time()
                # Only store final formatted turns
                if event.end_of_turn and event.turn_is_formatted:
                    transcripts[event.turn_order] = event.transcript
                    final_transcript_time[0] = time.time()

        def on_error(client, error):
            error_occurred[0] = error
            session_done.set()

        def on_terminated(client, event):
            session_done.set()

        # Create new session for each sample
        client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
            )
        )
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Error, on_error)
        client.on(StreamingEvents.Termination, on_terminated)

        # Connect (not counted in processing time)
        client.connect(StreamingParameters(sample_rate=16000, format_turns=True))

        # Start timing after connection is established
        stream_start_time[0] = time.time()

        # Stream audio in chunks
        chunk_size = 3200  # 100ms of 16kHz 16-bit audio
        for i in range(0, len(pcm_data), chunk_size):
            client.stream(pcm_data[i : i + chunk_size])
            time.sleep(0.02)

        # Record time when last audio chunk was sent
        audio_done_time = time.time()

        # Terminate session and wait for final transcripts
        client.disconnect(terminate=True)
        session_done.wait(timeout=30)

        # Calculate timing metrics
        ttfb = (
            (first_transcript_time[0] - stream_start_time[0]) if first_transcript_time[0] else None
        )
        # Processing time = time from last audio chunk sent to final transcript received
        # Can be negative if transcript arrives before we finish sending (real-time processing)
        if final_transcript_time[0]:
            processing_time = final_transcript_time[0] - audio_done_time
            # If negative, final transcript arrived before audio finished - set to 0
            processing_time = max(0, processing_time)
        else:
            processing_time = 0

        # Store for aggregation
        self.processing_times.append(processing_time)
        if ttfb is not None:
            self.ttfb_times.append(ttfb)

        # Print timing info
        ttfb_str = f"{ttfb * 1000:.0f}ms" if ttfb else "N/A"
        if processing_time > 0:
            processing_str = f"{processing_time * 1000:.0f}ms"
        else:
            processing_str = "0ms (real-time)"
        print(f"  [Streaming] TTFB: {ttfb_str}, Finalization: {processing_str}")

        if error_occurred[0]:
            raise RuntimeError(f"Streaming error: {error_occurred[0]}")

        full_transcript = " ".join(transcripts[k] for k in sorted(transcripts.keys()))
        # Return total elapsed time (stream start to final transcript) for consistency with other evaluators
        total_elapsed = (
            (final_transcript_time[0] - stream_start_time[0]) if final_transcript_time[0] else 0
        )
        return full_transcript, total_elapsed

    def compute_metrics(self) -> dict:
        """Compute final metrics including streaming-specific timing."""
        metrics = super().compute_metrics()
        # Add streaming timing stats
        if self.ttfb_times:
            metrics["avg_ttfb"] = sum(self.ttfb_times) / len(self.ttfb_times)
            metrics["min_ttfb"] = min(self.ttfb_times)
            metrics["max_ttfb"] = max(self.ttfb_times)
        if self.processing_times:
            metrics["avg_processing"] = sum(self.processing_times) / len(self.processing_times)
        return metrics


class DeepgramEvaluator(Evaluator):
    """Evaluator for Deepgram Nova 3 API."""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        from deepgram import DeepgramClient

        self.client = DeepgramClient(api_key=api_key)

    def transcribe(self, audio) -> tuple[str, float]:
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()

        response = self.client.listen.v1.media.transcribe_file(
            request=wav_bytes,
            model="nova-3",
        )
        elapsed = time.time() - start

        text = response.results.channels[0].alternatives[0].transcript
        return text, elapsed
