"""tiny-audio in one file: a speech LLM you can read in one sitting.

    audio -> speech encoder (frozen) -> projector (trained) -> LLM decoder (frozen) -> text

That is the whole idea. A pretrained encoder already hears; a pretrained LLM
already writes. Neither understands the other's vectors, so we train a small
MLP -- the projector -- to translate encoder frames into vectors the LLM
reads as if they were word embeddings. Only the projector learns. Everything
else is plumbing, and this file keeps the plumbing short.

It is deliberately simple: one clip at a time at inference, no kernels, no
LoRA, no mixed precision. The recipes that produced the published model live
in tiny_audio/ and scripts/train.py and do all of that properly; read this
file first, then those.

Run it:

    python model.py            # overfit 73 LibriSpeech clips, then transcribe a few

Defaults: Whisper-small's encoder and Qwen3-0.6B. Both are swappable.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    WhisperFeatureExtractor,
    WhisperModel,
)

ENCODER_ID = "openai/whisper-small"
DECODER_ID = "Qwen/Qwen3-0.6B"
SAMPLE_RATE = 16_000
INSTRUCTION = "Transcribe the audio."
AUDIO_SLOT = "<|audio|>"  # marks where audio goes in the prompt; never tokenized
IGNORE = -100  # label value the LLM's loss skips

Waveform = npt.NDArray[np.float32]  # mono, 16 kHz


# -----------------------------------------------------------------------------
# The projector: the only trained part.
# -----------------------------------------------------------------------------


class Projector(nn.Module):
    """Turn encoder frames into LLM "words".

    Whisper emits one frame every 20 ms: 50 frames per second, far more than an
    LLM needs. So we first glue k neighbouring frames into one long vector
    (frame stacking, k=4 -> 80 ms per audio token), then a two-layer MLP maps
    that vector into the LLM's embedding space. Same layout as
    tiny_audio/projectors.py, minus the configuration.
    """

    def __init__(self, encoder_dim: int, llm_dim: int, k: int = 4, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.k = k
        self.norm = nn.RMSNorm(encoder_dim * k, eps=1e-6)
        self.linear_1 = nn.Linear(encoder_dim * k, hidden_dim)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_dim, llm_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """[batch, frames, encoder_dim] -> [batch, frames // k, llm_dim]."""
        batch, n_frames, dim = frames.shape
        n_tokens = n_frames // self.k  # drop the ragged tail
        stacked = frames[:, : n_tokens * self.k].reshape(batch, n_tokens, dim * self.k)
        return self.linear_2(self.act(self.linear_1(self.norm(stacked))))


# -----------------------------------------------------------------------------
# The model: encoder + projector + decoder, and how their inputs line up.
# -----------------------------------------------------------------------------


class TinyAudio(nn.Module):
    """Frozen encoder, trained projector, frozen decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        feature_extractor: WhisperFeatureExtractor,
        encoder_dim: int,
        k: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.projector = Projector(encoder_dim, decoder.config.hidden_size, k=k)
        # Freeze the two pretrained halves: gradients flow *through* the
        # decoder to the projector, but no decoder weight changes.
        for param in [*encoder.parameters(), *decoder.parameters()]:
            param.requires_grad = False

        # The prompt is the decoder's own chat format, so a frozen instruct
        # model sees something familiar. We render it once, then split it
        # around the audio slot: [before] [audio tokens] [after] [transcript].
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{AUDIO_SLOT}{INSTRUCTION}"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: skip the <think> block; others ignore it
        )
        before, after = str(prompt).split(AUDIO_SLOT)
        self.before_ids: list[int] = tokenizer.encode(before, add_special_tokens=False)
        self.after_ids: list[int] = tokenizer.encode(after, add_special_tokens=False)

    @classmethod
    def from_pretrained(cls, encoder_id: str = ENCODER_ID, decoder_id: str = DECODER_ID):
        """Download both halves from the Hugging Face Hub and add a fresh projector."""
        # float32 throughout: slower and bigger than bf16, but nothing to debug.
        whisper = WhisperModel.from_pretrained(encoder_id, dtype=torch.float32)
        return cls(
            encoder=whisper.encoder,  # keep the encoder, drop Whisper's own decoder
            decoder=AutoModelForCausalLM.from_pretrained(decoder_id, dtype=torch.float32),
            tokenizer=AutoTokenizer.from_pretrained(decoder_id),
            feature_extractor=WhisperFeatureExtractor.from_pretrained(encoder_id),
            encoder_dim=whisper.config.d_model,
        )

    @property
    def device(self) -> torch.device:
        """Where the weights live."""
        return self.projector.linear_1.weight.device

    def embed_audio(self, audios: list[Waveform]) -> list[torch.Tensor]:
        """16 kHz waveforms -> one [n_audio_tokens, llm_dim] tensor per clip."""
        # 1. Waveform -> log-mel spectrogram, one frame per 10 ms. Whisper
        #    always pads to 30 s, so remember how much of each clip is real.
        features = self.feature_extractor(
            audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
        )
        mel = features.input_features.to(self.device)
        real_mel_frames = features.attention_mask.sum(dim=-1)

        # 2. Spectrogram -> encoder frames. A stride-2 conv halves the rate,
        #    so one encoder frame per 20 ms: [batch, 1500, encoder_dim].
        with torch.no_grad():
            frames = self.encoder(mel).last_hidden_state

        # 3. Encoder frames -> LLM-space audio tokens: [batch, 375, llm_dim].
        tokens = self.projector(frames)

        # 4. Cut each clip's padding back off, so a 3 s clip costs ~37 audio
        #    tokens instead of 375.
        keep = (real_mel_frames // 2 // self.projector.k).clamp(min=1)
        return [tokens[i, : int(n)] for i, n in enumerate(keep)]

    def embed_text(self, ids: list[int]) -> torch.Tensor:
        """Token ids -> the decoder's own word embeddings."""
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        return self.decoder.get_input_embeddings()(ids_tensor)

    def build_prompt(
        self, audio: torch.Tensor, transcript: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lay out one example as embeddings [seq, llm_dim] and labels [seq].

        The labels are IGNORE everywhere except the transcript, so the loss
        only asks the decoder to predict the words -- not the prompt, not the
        audio. With no transcript (inference) the prompt ends where the
        decoder should start writing.
        """
        parts = [self.embed_text(self.before_ids), audio, self.embed_text(self.after_ids)]
        n_prompt = sum(len(part) for part in parts)
        target: list[int] = []
        if transcript is not None:
            eos = self.tokenizer.eos_token_id
            assert isinstance(eos, int)
            # Ending on the end-of-turn token teaches the decoder when to stop.
            target = [*self.tokenizer.encode(transcript, add_special_tokens=False), eos]
            parts.append(self.embed_text(target))
        labels = [IGNORE] * n_prompt + target
        return torch.cat(parts), torch.tensor(labels, device=self.device)

    def forward(self, audios: list[Waveform], transcripts: list[str]) -> torch.Tensor:
        """Next-token loss on the transcripts, given the audio."""
        examples = [
            self.build_prompt(audio, text)
            for audio, text in zip(self.embed_audio(audios), transcripts, strict=True)
        ]
        # Right-pad to one length; padded positions get attention 0 and IGNORE.
        seq = max(len(embeds) for embeds, _ in examples)
        dim = examples[0][0].shape[-1]
        inputs = torch.zeros(len(examples), seq, dim, device=self.device)
        attention = torch.zeros(len(examples), seq, dtype=torch.long, device=self.device)
        labels = torch.full((len(examples), seq), IGNORE, device=self.device)
        for i, (embeds, example_labels) in enumerate(examples):
            inputs[i, : len(embeds)] = embeds
            attention[i, : len(embeds)] = 1
            labels[i, : len(embeds)] = example_labels
        # The decoder shifts labels by one and averages cross-entropy itself.
        return self.decoder(inputs_embeds=inputs, attention_mask=attention, labels=labels).loss

    @torch.inference_mode()
    def transcribe(self, audio: Waveform, max_new_tokens: int = 128) -> str:
        """Greedy-decode one clip."""
        embeds, _ = self.build_prompt(self.embed_audio([audio])[0])
        # PreTrainedModel's stubs don't declare generate(); it comes from GenerationMixin.
        out: torch.Tensor = self.decoder.generate(  # pyright: ignore[reportCallIssue]
            inputs_embeds=embeds[None],
            attention_mask=torch.ones(1, len(embeds), dtype=torch.long, device=self.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        # Given only embeddings, generate() returns just the new tokens.
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        assert isinstance(text, str)  # one sequence in, one string out
        return text.strip()


# -----------------------------------------------------------------------------
# A training loop small enough to watch: overfit a handful of clips.
# -----------------------------------------------------------------------------


def main(steps: int = 300, batch_size: int = 8, lr: float = 1e-3) -> None:
    """Train the projector on LibriSpeech's 73-clip dummy split, then transcribe."""
    from datasets import Audio, Dataset, load_dataset  # noqa: PLC0415  (only needed here)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    model = TinyAudio.from_pretrained().to(device)
    data = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    assert isinstance(data, Dataset)  # one split, fully loaded
    data = data.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    rows: list[dict[str, Any]] = data.to_list()
    audios = [np.asarray(row["audio"]["array"], dtype=np.float32) for row in rows]
    texts = [str(row["text"]).capitalize() for row in rows]

    # Only the projector's parameters go to the optimizer.
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.projector.parameters())
    print(f"training {n_params / 1e6:.1f}M projector parameters on {device}")

    rng = np.random.default_rng(0)
    for step in range(steps):
        batch = rng.choice(len(audios), size=batch_size, replace=False)
        loss = model([audios[i] for i in batch], [texts[i] for i in batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss {loss.item():.3f}")

    for i in range(3):
        print(f"\nreference: {texts[i]}\nmodel:     {model.transcribe(audios[i])}")


if __name__ == "__main__":
    main()
