"""Configuration for the ASR model: encoder, decoder, projector and training options."""

import transformers

# Default conv layers for Whisper/GLM-ASR audio encoders: [(pad, kernel, stride), ...]
DEFAULT_ENCODER_CONV_LAYERS = [(1, 3, 1), (1, 3, 2)]

# Granite Speech 5.0 TurboCTC. Its feature extractor already stacks mel frame
# pairs (100 Hz -> 50 Hz, 320-dim output), so the `mel_length` handed to
# compute_encoder_output_length is the 50 Hz frame count and only the encoder's
# own 4x reduction remains. Granite subsamples with `t // 2` (floor, dropping a
# trailing odd frame) in each of the two `subsample_layers`; two
# (pad=0, kernel=2, stride=2) entries reproduce floor(L/2) exactly under the
# generic formula below. Verified against the real encoder at 1/2/5/10/20s
# (T_fe 50/100/250/500/1000 -> 12/25/62/125/250).
# Full path: 100 Hz mel -> 2x FE stacking -> 4x encoder -> 12.5 Hz.
GRANITE_ENCODER_CONV_LAYERS = [(0, 2, 2), (0, 2, 2)]

# Encoders whose `input_features` are (batch, time, feature_dim) rather than
# Whisper/GLM-ASR's (batch, n_mels, mel_len). Conformer-family checkpoints
# (Granite Speech 5.0, Parakeet, Nemotron) are all time-major.
_TIME_MAJOR_ENCODER_MARKERS = ("granite-speech", "parakeet", "nemotron")

# Decoders that already ship a native audio-placeholder token. Reusing it is
# strictly better than adding "<audio>" and resizing, above all when the
# decoder is frozen: masked_scatter overwrites the placeholder's *input*
# embedding, but Gemma4 also looks the token up in `embed_tokens_per_layer` to
# build its per-layer embeddings (PLE), and that row is NOT overwritten. A
# freshly added row would hand every audio position a random, permanently
# untrainable PLE vector. Gemma's own "<|audio|>" row is pretrained for
# exactly this placeholder role.
_NATIVE_AUDIO_TOKENS = {"gemma-4": "<|audio|>"}


def native_audio_token(text_model_id: str | None) -> str | None:
    """Return the decoder's built-in audio placeholder token, if it has one."""
    lowered = (text_model_id or "").lower()
    return next((tok for key, tok in _NATIVE_AUDIO_TOKENS.items() if key in lowered), None)


def is_time_major_encoder(audio_model_id: str | None) -> bool:
    """Whether `audio_model_id` expects (batch, time, feature) input_features."""
    return any(m in (audio_model_id or "").lower() for m in _TIME_MAJOR_ENCODER_MARKERS)


def compute_encoder_output_length(mel_length, conv_layers=None):
    """Apply encoder conv layer formulas to compute output length.

    Works with both Python ints and torch tensors of mel lengths; the formula
    `(L + 2*p - (k-1) - 1) // s + 1` per layer is identical for both.
    """
    layers = conv_layers if conv_layers is not None else DEFAULT_ENCODER_CONV_LAYERS
    length = mel_length
    for padding, kernel_size, stride in layers:
        length = (length + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    return length


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
        encoder_conv_layers: list | None = None,
        audio_sample_rate: int = 16000,
        # Whether the encoder takes `input_features` as (batch, time, feature)
        # instead of Whisper/GLM-ASR's (batch, n_mels, mel_len). Only
        # SpecAugment unpacks all three dims, so this exists to keep
        # `_mask_input_features` masking the time axis rather than the feature
        # axis. Left as None it is auto-detected from `audio_model_id`, so a
        # Granite/Parakeet swap can't silently mask the wrong axis.
        audio_features_time_major: bool | None = None,
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
        # Defaults to the decoder's native audio token when it has one (Gemma 4),
        # otherwise "<audio>", which is added to the tokenizer and requires an
        # embedding resize.
        audio_token: str | None = None,
        # dtype for the projector alone. The fp32-master-weights argument only
        # applies to parameters an optimizer actually updates, so pinning the
        # whole stack to float32 to protect a 10M-param projector wastes 2
        # bytes on every frozen parameter -- 10.4 GiB on a frozen Gemma 4 E2B
        # (5.12B stored params; "E2B" counts *active* params, and the
        # per-layer-embedding table alone is 2.35B). Set this to float32 while
        # model_dtype stays bfloat16 to get master-weight precision where it
        # matters at frozen-model memory cost. Defaults to model_dtype, so
        # existing recipes are unchanged.
        projector_dtype: str | None = None,
        projector_pool_stride: int = 4,
        projector_hidden_dim: int | None = None,
        projector_type: str = "mlp",
        # Target RMS for the projector's output, i.e. the magnitude at which
        # audio tokens enter the decoder's residual stream. "auto" measures the
        # decoder's own `embed_tokens` RMS at construction; a float sets it
        # directly; None leaves PyTorch's default init alone.
        #
        # This is a property of the DECODER, not the projector. Qwen/Llama/
        # Mistral embed plainly, so their tokens sit near 0.015. Gemma-family
        # decoders multiply embeddings by sqrt(hidden_size), so theirs sit near
        # 1.0 -- a ~67x difference. Hardcoding either number breaks the other,
        # hence "auto".
        #
        # Why it matters: the decoder is pre-norm, so every sublayer reads a
        # normalized copy and the loss is nearly blind to prefix magnitude. But
        # the residual stream is NOT normalized, so audio tokens entering at
        # 13x the text scale make each sublayer's contribution at those
        # positions 13x smaller in relative terms -- the prefix passes through
        # all layers close to unchanged. Measured at init on granite_qwen
        # before this landed: projector out 0.198 vs embed_tokens 0.0150.
        projector_output_rms: float | str | None = "auto",
        # Label smoothing applied inside the LM's loss function (not HF Trainer's
        # LabelSmoother). Train-only — ASRModel.forward zeros it on eval. Routing
        # smoothing through the loss_function flows through liger's fused linear
        # CE when apply_liger_kernel_to_qwen3() is active, avoiding the
        # (B,T,V) fp32 log_softmax materialization that the HF LabelSmoother
        # path requires (~15GB at B=50/V=152k on Qwen3-0.6B).
        label_smoothing: float = 0.0,
        # LoRA configuration (for Stage 2 fine-tuning)
        use_lora: bool = False,
        lora_rank: int = 8,  # SALMONN default
        lora_alpha: int = 32,  # SALMONN default (scaling factor 4.0)
        lora_dropout: float = 0.0,
        lora_target_modules: list | None = None,  # Default: all linear layers
        freeze_projector: bool = False,  # True for Stage 2 (LoRA-only training)
        freeze_language_model: bool = True,  # False = full decoder fine-tuning
        freeze_text_embed_tokens: bool = False,
        # Audio encoder is frozen by default — the published recipe treats
        # GLM-ASR-Nano as a fixed feature extractor. Setting this to False
        # makes the encoder trainable; pair with `encoder_learning_rate` in
        # the training config to avoid destroying pretrained encoder weights
        # at the projector/decoder LR.
        freeze_audio_encoder: bool = True,
        # SpecAugment on mel input (training-only), parameters match
        # transformers' WhisperConfig / Wav2Vec2 conventions. Most relevant
        # when the encoder is trainable (`freeze_audio_encoder=False`) —
        # without augmentation the encoder sees identical mel inputs on
        # every visit and overfits fast. Standard for ASR encoder fine-
        # tuning (Whisper, Conformer, wav2vec2 all use it). Applied to
        # log-mel input where zero is in-distribution (silence);
        # structurally different from the prior encoder-output ZM which
        # was removed because zero was OOD for the encoder's emission
        # distribution. Uses `_compute_mask_indices` from
        # transformers.models.whisper.modeling_whisper — the same helper
        # Whisper itself uses, vectorized over the batch and torch.compile
        # compatible. Default values match Whisper's defaults.
        apply_spec_augment: bool = False,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,
        max_new_tokens: int | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        """Initialize ASR model configuration.

        Args:
            audio_model_id: HuggingFace model ID for audio encoder (GLM-ASR/Whisper)
            text_model_id: HuggingFace model ID for text decoder (Qwen)
            attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")
            model_dtype: Model dtype ("bfloat16", "float16", "float32")
            projector_type: Projector architecture ("mlp")
            use_lora: Enable LoRA adapters for Stage 2 fine-tuning
        """
        # Set default generation parameters (greedy decoding only).
        # Applied via setattr below — keeping these out of kwargs so they
        # don't get re-overwritten by super().__init__(**kwargs) at the end.
        generation_defaults = {
            "max_new_tokens": 128,
            "use_cache": True,
        }

        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id
        self.attn_implementation = attn_implementation
        self.model_dtype = model_dtype
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.encoder_conv_layers = encoder_conv_layers or DEFAULT_ENCODER_CONV_LAYERS
        self.audio_features_time_major = (
            is_time_major_encoder(audio_model_id)
            if audio_features_time_major is None
            else audio_features_time_major
        )
        self.encoder_attention_mask = (
            is_time_major_encoder(audio_model_id)
            if encoder_attention_mask is None
            else encoder_attention_mask
        )
        self.audio_token = audio_token or native_audio_token(text_model_id) or "<audio>"
        self.projector_dtype = projector_dtype or model_dtype
        self.audio_sample_rate = audio_sample_rate
        self.projector_pool_stride = projector_pool_stride
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_type = projector_type
        self.projector_output_rms = projector_output_rms
        self.transcribe_prompt = transcribe_prompt
        self.label_smoothing = label_smoothing
        # LoRA configuration
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
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
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks

        explicit_generation_args = {
            "max_new_tokens": max_new_tokens,
            "use_cache": use_cache,
        }
        for key, default in generation_defaults.items():
            value = explicit_generation_args[key]
            setattr(self, key, value if value is not None else default)

        if "audio_config" not in kwargs:
            self.audio_config = transformers.AutoConfig.from_pretrained(audio_model_id)
            # Override dtype to match model_dtype
            self.audio_config.dtype = model_dtype
        else:
            self.audio_config = kwargs.pop("audio_config")

        if "text_config" not in kwargs:
            self.text_config = transformers.AutoConfig.from_pretrained(
                text_model_id, trust_remote_code=True
            )
            # Override dtype to match model_dtype
            self.text_config.dtype = model_dtype
        else:
            self.text_config = kwargs.pop("text_config")

        if isinstance(self.text_config, dict):
            # Reconstruct config from dict using the model_type stored in the dict
            model_type = self.text_config["model_type"]
            config_class = transformers.AutoConfig.for_model(model_type).__class__
            self.text_config = config_class(**self.text_config)

        if isinstance(self.audio_config, dict):
            model_type = self.audio_config.get("model_type")
            if model_type:
                config_class = transformers.AutoConfig.for_model(model_type).__class__
                self.audio_config = config_class(**self.audio_config)

        super().__init__(**kwargs)

        # Point encoder to audio_config so pipeline uses correct feature extractor
        # The pipeline looks for config.encoder._name_or_path for feature extractor
        self.encoder = self.audio_config

        self.auto_map = {
            "AutoConfig": "asr_config.ASRConfig",
            "AutoModel": "asr_modeling.ASRModel",
            "AutoModelForSpeechSeq2Seq": "asr_modeling.ASRModel",
            "AutoProcessor": "asr_processing.ASRProcessor",
        }
        self.custom_pipelines = {
            "automatic-speech-recognition": {
                "impl": "asr_pipeline.ASRPipeline",
                "pt": ["AutoModelForSpeechSeq2Seq"],
                "tf": [],
                "type": "audio",
            }
        }
        self.architectures = ["ASRModel"]
        self.pipeline_tag = "automatic-speech-recognition"


transformers.AutoConfig.register("asr_model", ASRConfig)
