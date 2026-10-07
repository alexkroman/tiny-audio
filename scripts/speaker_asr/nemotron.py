"""Speaker-attributed transcription: stock Qwen3-ASR words, Nemotron speakers.

    result = transcribe_diarized(asr, asr_processor, diarizer, audio_16k)
    result.text      # '<SPK_1>...<SPK_2>...' over the whole recording
    result.turns     # [Turn(speaker, text, start, end), ...]

Transcribe-then-assign, not diarize-then-transcribe. The recording is
transcribed in chunks of at most `chunk_s` by an untouched Qwen3-ASR, every
word gets a time from forced alignment (tiny_audio.alignment.ForcedAligner),
and each word goes to the speaker Nemotron-3-Diarization finds most active over
the word's span. Cutting the audio at Nemotron's turns and transcribing each
one instead would transcribe overlapped speech twice and hand the ASR short,
context-poor pieces. Here WER is exactly the stock model's, so the attribution
gap (cpWER - WER) measures the diarizer and the word assignment alone.

Nemotron runs once over the whole recording in its offline mode, which chunks
internally (27.2 s + 3.2 s look-ahead) and carries a speaker cache across
chunks, so speaker ids are recording-level with no linking step. It tracks at
most 8 speakers. Needs a transformers build with `nemotron3_diarization`
(main as of 2026-10; not in 5.17).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from scripts.speaker_asr.longform import SAMPLE_RATE, LongFormResult, Turn, chunk_bounds

DIARIZER_ID = "nvidia/Nemotron-3-Diarization"
FRAME_S = 0.01  # Nemotron emits one logit row per 10 ms


@dataclass
class Diarizer:
    model: torch.nn.Module
    processor: object

    @classmethod
    def load(cls, model_id: str = DIARIZER_ID, device: str | None = None) -> Diarizer:
        import importlib.util

        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        from tiny_audio.turns.model import pick_device

        if importlib.util.find_spec("transformers.models.nemotron3_diarization") is None:
            raise ImportError(
                "Nemotron-3-Diarization needs a transformers build with `nemotron3_diarization` "
                "(main as of 2026-10): pip install git+https://github.com/huggingface/transformers"
            )

        device = device or pick_device()
        model = AutoModelForAudioFrameClassification.from_pretrained(model_id).to(device).eval()
        return cls(model, AutoProcessor.from_pretrained(model_id))

    @torch.inference_mode()
    def activity(self, audio: np.ndarray) -> np.ndarray:
        """(frames, speakers) speech probabilities at 10 ms; columns in arrival order."""
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        return self.model(**inputs).logits[0].float().sigmoid().cpu().numpy()


ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"


@dataclass
class QwenAligner:
    """Qwen3-ForcedAligner behind ForcedAligner.align's interface.

    Returns `[{"word", "start", "end"}]` for the words of `text` it can align,
    in order, with the ORIGINAL word (punctuation, casing) as "word" so
    `time_all_words` can match it back. The aligner itself sees words with
    punctuation stripped and drops any that strip to nothing ("--"); this
    mirrors that split word by word to keep the pairing exact.
    """

    model: torch.nn.Module
    processor: object

    @classmethod
    def load(cls, model_id: str = ALIGNER_ID, device: str | None = None) -> QwenAligner:
        from transformers import AutoProcessor, Qwen3ASRForTokenClassification

        from tiny_audio.turns.model import pick_device

        device = device or pick_device()
        model = Qwen3ASRForTokenClassification.from_pretrained(model_id, dtype=torch.bfloat16)
        return cls(model.to(device).eval(), AutoProcessor.from_pretrained(model_id))

    @torch.inference_mode()
    def align(self, audio: np.ndarray, text: str) -> list[dict]:
        from transformers.models.qwen3_asr.processing_qwen3_asr import _clean_tokens

        kept = [w for w in text.split() if _clean_tokens([w])]
        if not kept:
            return []
        inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio, " ".join(kept), language="English", return_tensors="pt"
        )
        inputs = {
            k: (
                v.to(self.model.device, dtype=self.model.dtype)
                if v.is_floating_point()
                else v.to(self.model.device)
            )
            for k, v in inputs.items()
        }
        logits = self.model(**inputs).logits
        (timed,) = self.processor.decode_forced_alignment(
            logits, inputs["input_ids"], word_lists, self.model.config.timestamp_token_id
        )
        return [
            {"word": word, "start": t["start_time"], "end": t["end_time"]}
            for word, t in zip(kept, timed, strict=True)
        ]


def assign_speakers(
    words: list[tuple[float | None, float | None]],
    activity: np.ndarray,
    min_activity: float = 0.1,
) -> list[int | None]:
    """Speaker column per word: the one with the highest mean activity over its span.

    A word with no time, or whose best speaker never reaches `min_activity`
    (Nemotron heard nobody there), is left None for `fill_gaps`.
    """
    out: list[int | None] = []
    for start, end in words:
        if start is None or end is None:
            out.append(None)
            continue
        lo = min(int(start / FRAME_S), len(activity) - 1)
        hi = max(lo + 1, min(int(np.ceil(end / FRAME_S)), len(activity)))
        mean = activity[lo:hi].mean(axis=0)
        best = int(mean.argmax())
        out.append(best if mean[best] >= min_activity else None)
    return out


def fill_gaps(speakers: list[int | None]) -> list[int]:
    """Unassigned words join the previous speaker (the next one at the start)."""
    known = [s for s in speakers if s is not None]
    if not known:
        return [0] * len(speakers)
    filled, last = [], known[0]
    for s in speakers:
        last = s if s is not None else last
        filled.append(last)
    return filled


def time_all_words(text: str, aligned: list[dict], offset: float) -> list[tuple]:
    """(word, start, end) for EVERY word of `text`; None times where alignment skipped it.

    ForcedAligner drops words with no alignable characters ("25", "--"). They
    still count toward WER, so they stay in the transcript with no time
    instead of being dropped (unlike longform.time_words).
    """
    out, j = [], 0
    for word in text.split():
        if j < len(aligned) and aligned[j]["word"] == word:
            out.append((word, aligned[j]["start"] + offset, aligned[j]["end"] + offset))
            j += 1
        else:
            out.append((word, None, None))
    return out


def group_turns(words: list[tuple], speakers: list[int]) -> list[Turn]:
    """Consecutive words of one speaker become one turn."""
    turns: list[Turn] = []
    for (word, start, end), speaker in zip(words, speakers, strict=True):
        if turns and turns[-1].speaker == speaker:
            turn = turns[-1]
            turn.text += " " + word
            turn.end = end if end is not None else turn.end
        else:
            turns.append(Turn(speaker, word, start, end))
    return turns


def transcribe_diarized(
    asr_model,
    asr_processor,
    diarizer: Diarizer,
    audio: np.ndarray,
    chunk_s: float = 25.0,
    min_chunk_s: float = 10.0,
    min_activity: float = 0.1,
    align=None,
    max_new_tokens: int = 320,
) -> LongFormResult:
    """Speaker-attributed transcript of 16 kHz mono `audio` of any length."""
    from scripts.speaker_asr.model import transcribe_speakers

    if align is None:
        from tiny_audio.alignment import ForcedAligner

        align = ForcedAligner.align
    audio = np.asarray(audio, dtype=np.float32)
    activity = diarizer.activity(audio)

    words: list[tuple] = []
    bounds = chunk_bounds(audio, chunk_s, min_chunk_s)
    for s, e in bounds:
        live = audio[s:e]
        # n_speakers=0: a stock model has no speaker tokens, so this is plain text.
        (text,) = transcribe_speakers(asr_model, asr_processor, [live], 0, max_new_tokens)
        if text.strip():
            words += time_all_words(text, align(live, text), s / SAMPLE_RATE)

    speakers = fill_gaps(assign_speakers([(a, b) for _, a, b in words], activity, min_activity))
    return LongFormResult(
        group_turns(words, speakers), [(s / SAMPLE_RATE, e / SAMPLE_RATE) for s, e in bounds]
    )
