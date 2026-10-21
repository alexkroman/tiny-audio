"""Configuration for the ASR model: encoder, decoder, projector and training options."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, overload

import transformers

if TYPE_CHECKING:
    import torch

ConvLayerSpec = tuple[int, int, int]

# Default conv layers for Whisper/GLM-ASR audio encoders: [(pad, kernel, stride), ...]
DEFAULT_ENCODER_CONV_LAYERS: list[ConvLayerSpec] = [(1, 3, 1), (1, 3, 2)]

# Granite Speech 5.0 TurboCTC. Its feature extractor already stacks mel frame
# pairs (100 Hz -> 50 Hz, 320-dim output), so the `mel_length` handed to
# compute_encoder_output_length is the 50 Hz frame count and only the encoder's
# own 4x reduction remains. Granite subsamples with `t // 2` (floor, dropping a
# trailing odd frame) in each of the two `subsample_layers`; two
# (pad=0, kernel=2, stride=2) entries reproduce floor(L/2) exactly under the
# generic formula below. Verified against the real encoder at 1/2/5/10/20s
# (T_fe 50/100/250/500/1000 -> 12/25/62/125/250).
# Full path: 100 Hz mel -> 2x FE stacking -> 4x encoder -> 12.5 Hz.
GRANITE_ENCODER_CONV_LAYERS: list[ConvLayerSpec] = [(0, 2, 2), (0, 2, 2)]

# The decoding knobs ASRConfig owns, with their defaults. ASRModel copies
# exactly these onto its GenerationConfig, so a new knob is one entry here.
# 12 is chosen to be inert on real speech: it blocks only exact 12-gram
# repeats, and natural English -- including genuine stutters, list
# recitation and "very very very" -- effectively never repeats a
# 12-token span verbatim. Measured rate of runaway loops is 1 in 10,630
# rows, so this should fire almost never; it bounds the tail rather
# than changing normal output.
GENERATION_DEFAULTS: dict[str, Any] = {
    "max_new_tokens": 128,
    "use_cache": True,
    "no_repeat_ngram_size": 12,
}

# Encoders whose `input_features` are (batch, time, feature_dim) rather than
# Whisper/GLM-ASR's (batch, n_mels, mel_len). Conformer-family checkpoints
# (Granite Speech 5.0, Parakeet, Nemotron) are all time-major.
_TIME_MAJOR_ENCODER_MARKERS = ("granite-speech", "parakeet", "nemotron")


def is_time_major_encoder(audio_model_id: str | None) -> bool:
    """Whether `audio_model_id` expects (batch, time, feature) input_features."""
    return any(m in (audio_model_id or "").lower() for m in _TIME_MAJOR_ENCODER_MARKERS)


# Qwen3-ASR's encoder (tiny_audio/qwen3_asr_encoder.py) runs its convolutions on
# independent 1 s chunks of 100 mel frames, three (pad 1, kernel 3, stride 2)
# layers each, so a full chunk yields 13 frames: transformers'
# `_get_feat_extract_output_lengths` for that model, restated here.
QWEN3_ASR_CHUNK_FRAMES = 100
QWEN3_ASR_CONV_LAYERS: list[ConvLayerSpec] = [(1, 3, 2), (1, 3, 2), (1, 3, 2)]


def is_qwen3_asr_encoder(audio_model_id: str | None) -> bool:
    """Whether `audio_model_id` names a Qwen3-ASR checkpoint (its encoder is used)."""
    return "qwen3-asr" in (audio_model_id or "").lower()


@overload
def compute_encoder_output_length(
    mel_length: int,
    conv_layers: Sequence[ConvLayerSpec] | None = None,
    chunk_frames: int | None = None,
) -> int: ...
@overload
def compute_encoder_output_length(
    mel_length: "torch.Tensor",
    conv_layers: Sequence[ConvLayerSpec] | None = None,
    chunk_frames: int | None = None,
) -> "torch.Tensor": ...
def compute_encoder_output_length(
    mel_length: "int | torch.Tensor",
    conv_layers: Sequence[ConvLayerSpec] | None = None,
    chunk_frames: int | None = None,
) -> "int | torch.Tensor":
    """Apply encoder conv layer formulas to compute output length.

    Works with both Python ints and torch tensors of mel lengths; the formula
    `(L + 2*p - (k-1) - 1) // s + 1` per layer is identical for both.

    With `chunk_frames`, the encoder convolves independent chunks of that many
    mel frames (Qwen3-ASR): every full chunk contributes the same count and the
    remainder its own, and an empty remainder contributes zero.
    """
    layers = conv_layers if conv_layers is not None else DEFAULT_ENCODER_CONV_LAYERS
    if chunk_frames:
        per_chunk = compute_encoder_output_length(chunk_frames, layers)
        rest = compute_encoder_output_length(mel_length % chunk_frames, layers)
        return (mel_length // chunk_frames) * per_chunk + rest
    length = mel_length
    for padding, kernel_size, stride in layers:
        length = (length + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    return length


def _config_from_dict(model_type: str, values: dict[str, Any]) -> transformers.PretrainedConfig:
    """Rebuild a serialized sub-config as the config class registered for `model_type`."""
    # Serialized configs carry their own `model_type`, which would collide with
    # for_model's positional argument; the config class already sets it.
    kwargs = {key: value for key, value in values.items() if key != "model_type"}
    return transformers.AutoConfig.for_model(model_type, **kwargs)


class ASRConfig(transformers.PretrainedConfig):
    """Configuration class for the ASR model.

    This config combines settings for:
    - Audio encoder (GLM-ASR/Whisper)
    - Text decoder (Qwen)
    - Projector (MLP)
    - Generation parameters
    - Training options (LoRA)
    """

    model_type = "asr_model"
    is_composition = True

    def __init__(
        self,
        audio_model_id: str = "zai-org/GLM-ASR-Nano-2512",
        text_model_id: str = "Qwen/Qwen3-0.6B",
        attn_implementation: str = "flash_attention_2",
        model_dtype: str = "bfloat16",
        # Instruction appended after the audio placeholders at inference. Must
        # match the prompt the desired output convention was TRAINED under --
        # when a run splits prompts by whether a source is punctuated (see
        # `text_punct` in the data configs), inference has to pick the half it
        # actually wants, which for a formatted-text eval is the punctuated one.
        # None keeps ASRModel.TRANSCRIBE_PROMPT's class default.
        transcribe_prompt: str | None = None,
        encoder_dim: int | None = None,
        llm_dim: int | None = None,
        # Encoder conv layers: list of (padding, kernel_size, stride) tuples
        # Default is Whisper/GLM-ASR structure: conv1(k=3,s=1,p=1) + conv2(k=3,s=2,p=1)
        encoder_conv_layers: list[ConvLayerSpec] | None = None,
        # Mel frames per independently convolved encoder chunk (Qwen3-ASR's
        # 100); None picks it from `audio_model_id`, 0 means no chunking.
        encoder_chunk_frames: int | None = None,
        # Pack labelled batches padding-free when the decoder runs FA2
        # (asr_packing.py). Ignored under SDPA/eager, where it would be wrong.
        pack_sequences: bool = True,
        # Longest chunk the pipeline cuts long audio into. Must stay under the
        # clip length the model trained on: a model trained on <=19 s skipped
        # and paraphrased 25-30 s chunks (E22 8.00 -> 9.41, 2026-10-09). 18 s
        # matches that 19 s training cap and is what older checkpoints, whose
        # config.json lacks this key, keep using.
        inference_chunk_seconds: float = 18.0,
        audio_sample_rate: int = 16000,
        # Whether to forward the mel padding mask into the audio encoder.
        # This matters a lot for Granite: it uses block attention over fixed
        # 128-frame blocks, so with `padding="longest"` batches the pad frames
        # leak into real frames -- measured max abs difference on the VALID
        # frames of a 1s clip padded alongside a 10s clip is 2.14.
        #
        # Auto-detected as True for the Conformer family and False for
        # Whisper/GLM-ASR. The legacy path is deliberately left unchanged:
        # GLM-ASR has been trained without an encoder mask for every run in
        # this repo's history, and silently switching it would make new runs
        # incomparable to those baselines. Flip it explicitly to test.
        encoder_attention_mask: bool | None = None,
        # Placeholder token whose embeddings get replaced by projector output.
        # Defaults to "<audio>", which is added to the tokenizer and requires an
        # embedding resize.
        audio_token: str | None = None,
        # dtype for the projector alone. The fp32-master-weights argument only
        # applies to parameters an optimizer actually updates, so pinning the
        # whole stack to float32 to protect a 10M-param projector wastes 2
        # bytes on every frozen parameter. Set this to float32 while
        # model_dtype stays bfloat16 to get master-weight precision where it
        # matters at frozen-model memory cost. Defaults to model_dtype, so
        # existing recipes are unchanged.
        projector_dtype: str | None = None,
        # Same idea as `projector_dtype`, for the encoder, and load-bearing the
        # moment `encoder_trainable_top_layers > 0`. AdamW allocates its states
        # with `zeros_like(param)`, so a bfloat16 encoder gets bfloat16 moments
        # and the update is applied in bfloat16. Granite Speech 5.0's weights
        # have RMS 0.0868, where one bf16 ULP is 4.88e-4; Adam's step at the
        # recommended encoder LR of 1e-5 is ~1e-5, i.e. 49x BELOW a single ULP,
        # so every update rounds away and the encoder silently does not train.
        # You would need lr ~5e-4 to clear the ULP, which is far too hot for a
        # pretrained Conformer. This is the same argument granite_qwen.yaml
        # makes for the decoder, and it bites harder here because the encoder's
        # weights are ~5x larger than the decoder's (0.087 vs 0.015-0.02).
        # Defaults to model_dtype, so frozen-encoder recipes are unchanged.
        encoder_dtype: str | None = None,
        projector_pool_stride: int = 4,
        projector_hidden_dim: int | None = None,
        # LoRA configuration (for Stage 2 fine-tuning)
        use_lora: bool = False,
        lora_rank: int = 8,  # SALMONN default
        lora_alpha: int = 32,  # SALMONN default (scaling factor 4.0)
        lora_dropout: float = 0.0,
        lora_target_modules: list[str] | None = None,  # Default: all linear layers
        # Seconds of silence prepended to every clip at INFERENCE. Measured on
        # Peoples (n=500, paired bootstrap): 20.51% -> 19.28% WER, delta -1.22
        # CI [-1.83, -0.64], with utterances losing a leading reference word
        # falling from 258/460 to 170/460. CommonVoice over the same protocol
        # is +0.30 CI [-0.43, +1.17] -- not significant -- so this is a win
        # where clips are cut mid-utterance and free where they are not.
        #
        # 0.25s, not more: the effect saturates there (0.50s gave no further
        # onset recovery). Training does NOT apply this -- the collator feeds
        # raw audio -- so it is an inference-only transform. Set 0.0 to disable.
        inference_lead_in_seconds: float = 0.25,
        freeze_projector: bool = False,  # True for Stage 2 (LoRA-only training)
        freeze_language_model: bool = True,  # False = full decoder fine-tuning
        freeze_text_embed_tokens: bool = False,
        # Audio encoder is frozen by default — the published recipe treats
        # GLM-ASR-Nano as a fixed feature extractor. Setting this to False
        # makes the encoder trainable; pair with `encoder_learning_rate` in
        # the training config to avoid destroying pretrained encoder weights
        # at the projector/decoder LR.
        freeze_audio_encoder: bool = True,
        # Partial encoder unfreeze: train only the TOP N transformer blocks
        # (plus the output projections that feed the projector), keeping the
        # lower blocks frozen. 0 = the all-or-nothing behaviour of
        # `freeze_audio_encoder` above.
        #
        # This is the middle option between "frozen feature extractor" and
        # "train all 472.9M encoder params on 7.6k hours". The lower blocks
        # carry general acoustic features learned from IBM's ~60k hours and
        # are the most expensive to damage; the top blocks are the ones
        # specialised to the encoder's own 16,384-BPE CTC head, which is
        # exactly the specialisation an LLM decoder does not want.
        #
        # Only meaningful with `freeze_audio_encoder: true` — this then
        # selectively re-enables the top N. Pair with `encoder_learning_rate`
        # (an order of magnitude below the projector LR).
        encoder_trainable_top_layers: int = 0,
        # Whether `encoder_trainable_top_layers` also unfreezes the post-stack
        # output projections (Granite's `out` / `out_mid`). True preserves the
        # original behaviour. False is worth it on Granite Speech 5.0, where
        # those two carry 33.57M params for 0.27% of the selection's squared
        # gradient -- see unfreeze_encoder_top_layers for the measurement.
        encoder_trainable_post_projections: bool = True,
        max_new_tokens: int | None = None,
        use_cache: bool | None = None,
        no_repeat_ngram_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ASR model configuration.

        Args:
            audio_model_id: HuggingFace model ID for audio encoder (GLM-ASR/Whisper)
            text_model_id: HuggingFace model ID for text decoder (Qwen)
            attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")
            model_dtype: Model dtype ("bfloat16", "float16", "float32")
            use_lora: Enable LoRA adapters for Stage 2 fine-tuning
        """
        # Set default generation parameters (greedy decoding only).
        # Applied via setattr below — keeping these out of kwargs so they
        # don't get re-overwritten by super().__init__(**kwargs) at the end.
        # `no_repeat_ngram_size` is the only non-greedy-neutral default here and
        # it is a guard, not a decoding strategy. Greedy decoding with
        # repetition_penalty=1.0 has no loop protection at all: one CommonVoice
        # sample emitted a correct transcript then repeated a 21-word phrase
        # four times until max_new_tokens cut it off, scoring 1513% on that row
        # and moving the 100-sample corpus WER from 8.30 to 31.83. The shipped
        # `_truncate_repetitions` postprocess missed it because its regex is
        # anchored to end-of-string and the loop was truncated mid-phrase.
        #
        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id
        self.attn_implementation = attn_implementation
        self.model_dtype = model_dtype
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.encoder_conv_layers: list[ConvLayerSpec] = (
            encoder_conv_layers or DEFAULT_ENCODER_CONV_LAYERS
        )
        self.pack_sequences: bool = pack_sequences
        self.inference_chunk_seconds: float = inference_chunk_seconds
        self.encoder_chunk_frames: int = (
            (QWEN3_ASR_CHUNK_FRAMES if is_qwen3_asr_encoder(audio_model_id) else 0)
            if encoder_chunk_frames is None
            else encoder_chunk_frames
        )
        # Qwen3-ASR's encoder takes the mel mask to drop padding itself.
        self.encoder_attention_mask: bool = (
            (is_time_major_encoder(audio_model_id) or is_qwen3_asr_encoder(audio_model_id))
            if encoder_attention_mask is None
            else encoder_attention_mask
        )
        self.audio_token: str = audio_token or "<audio>"
        self.projector_dtype: str = projector_dtype or model_dtype
        self.encoder_dtype: str = encoder_dtype or model_dtype
        self.audio_sample_rate = audio_sample_rate
        self.projector_pool_stride = projector_pool_stride
        self.projector_hidden_dim = projector_hidden_dim
        self.transcribe_prompt = transcribe_prompt
        # LoRA configuration
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.inference_lead_in_seconds: float = float(inference_lead_in_seconds)
        self.lora_target_modules: list[str] = lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        self.freeze_projector = freeze_projector
        self.freeze_language_model = freeze_language_model
        self.freeze_text_embed_tokens = freeze_text_embed_tokens
        self.freeze_audio_encoder = freeze_audio_encoder
        self.encoder_trainable_top_layers = encoder_trainable_top_layers
        self.encoder_trainable_post_projections = encoder_trainable_post_projections

        explicit_generation_args = {
            "max_new_tokens": max_new_tokens,
            "use_cache": use_cache,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        for key, default in GENERATION_DEFAULTS.items():
            value = explicit_generation_args[key]
            setattr(self, key, value if value is not None else default)

        audio_config: transformers.PretrainedConfig | dict[str, Any]
        if "audio_config" not in kwargs:
            audio_config = transformers.AutoConfig.from_pretrained(audio_model_id)
            # Override dtype to match model_dtype
            audio_config.dtype = model_dtype
        else:
            audio_config = kwargs.pop("audio_config")
        self.audio_config: transformers.PretrainedConfig | dict[str, Any] = audio_config

        text_config: transformers.PretrainedConfig | dict[str, Any]
        if "text_config" not in kwargs:
            text_config = transformers.AutoConfig.from_pretrained(
                text_model_id, trust_remote_code=True
            )
            # Override dtype to match model_dtype
            text_config.dtype = model_dtype
        else:
            text_config = kwargs.pop("text_config")

        if isinstance(text_config, dict):
            # Reconstruct config from dict using the model_type stored in the dict
            text_config = _config_from_dict(text_config["model_type"], text_config)
        self.text_config: transformers.PretrainedConfig = text_config

        if isinstance(audio_config, dict):
            audio_model_type = audio_config.get("model_type")
            if audio_model_type:
                self.audio_config = _config_from_dict(audio_model_type, audio_config)

        super().__init__(**kwargs)

        # Point encoder to audio_config so pipeline uses correct feature extractor
        # The pipeline looks for config.encoder._name_or_path for feature extractor
        self.encoder: transformers.PretrainedConfig | dict[str, Any] = self.audio_config

        # The Hub infers the model's task from the first model class listed, so
        # AutoModelForSpeechSeq2Seq must precede AutoModel (else: feature-extraction).
        self.auto_map: dict[str, str] = {
            "AutoConfig": "asr_config.ASRConfig",
            "AutoModelForSpeechSeq2Seq": "asr_modeling.ASRModel",
            "AutoModel": "asr_modeling.ASRModel",
            "AutoProcessor": "asr_processing.ASRProcessor",
        }
        self.custom_pipelines: dict[str, dict[str, Any]] = {
            "automatic-speech-recognition": {
                "impl": "asr_pipeline.ASRPipeline",
                "pt": ["AutoModelForSpeechSeq2Seq"],
                "tf": [],
                "type": "audio",
            }
        }
        self.architectures: list[str] | None = ["ASRModel"]
        self.pipeline_tag = "automatic-speech-recognition"


transformers.AutoConfig.register("asr_model", ASRConfig)
