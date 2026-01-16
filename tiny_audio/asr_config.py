from typing import Optional

import transformers


class ASRConfig(transformers.PretrainedConfig):
    """Configuration class for the ASR model.

    This config combines settings for:
    - Audio encoder (GLM-ASR/Whisper)
    - Text decoder (Qwen)
    - Projector (MLP, MOSA, MoE, QFormer)
    - Generation parameters
    - Training options (SpecAugment, LoRA)
    """

    model_type = "asr_model"
    is_composition = True

    def __init__(
        self,
        audio_model_id: str = "zai-org/GLM-ASR-Nano-2512",
        text_model_id: str = "Qwen/Qwen3-0.6B",
        attn_implementation: str = "flash_attention_2",
        model_dtype: str = "bfloat16",
        num_beams: Optional[int] = None,
        system_prompt: str = "You are a helpful assistant.",
        encoder_dim: Optional[int] = None,
        llm_dim: Optional[int] = None,
        # Encoder conv layers: list of (padding, kernel_size, stride) tuples
        # Default is Whisper/GLM-ASR structure: conv1(k=3,s=1,p=1) + conv2(k=3,s=2,p=1)
        encoder_conv_layers: Optional[list] = None,
        audio_sample_rate: int = 16000,
        projector_pool_stride: int = 4,
        downsample_rate: int = 5,  # Granite default
        projector_hidden_dim: Optional[int] = None,
        projector_type: str = "mlp",  # "mlp", "mosa", "moe", "qformer"
        projector_num_layers: int = 2,  # Number of layers in MLP projector
        projector_init_std: float = 0.02,  # Weight initialization std
        projector_dropout: float = 0.0,  # Dropout rate for projector layers
        # MoE-specific configuration
        num_experts: int = 4,  # Number of experts in MoE projectors
        num_experts_per_tok: int = 2,  # Top-k experts per token
        router_aux_loss_coef: float = 0.01,  # Auxiliary loss coefficient for load balancing
        # QFormer-specific configuration (Granite defaults)
        qformer_window_size: int = 15,  # Window size for QFormer processing
        qformer_hidden_size: Optional[int] = None,  # QFormer hidden size (defaults to encoder_dim)
        qformer_num_layers: int = 2,  # Number of QFormer transformer layers
        qformer_num_heads: int = 16,  # Number of attention heads in QFormer
        qformer_intermediate_size: Optional[int] = None,  # FFN size (defaults to 4x hidden)
        label_smoothing: float = 0.0,  # Label smoothing for cross-entropy loss
        inference_warmup_tokens: int = 10,
        # SpecAugment settings
        use_specaugment: bool = False,
        num_time_masks: int = 2,
        time_mask_length: int = 10,
        num_freq_masks: int = 0,
        freq_mask_length: int = 10,
        # LoRA configuration (for Stage 2 fine-tuning)
        use_lora: bool = False,
        lora_rank: int = 8,  # SALMONN default
        lora_alpha: int = 32,  # SALMONN default (scaling factor 4.0)
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,  # Default: all linear layers
        freeze_projector: bool = False,  # True for Stage 2 (LoRA-only training)
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize ASR model configuration.

        Args:
            audio_model_id: HuggingFace model ID for audio encoder (GLM-ASR/Whisper)
            text_model_id: HuggingFace model ID for text decoder (Qwen)
            attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")
            model_dtype: Model dtype ("bfloat16", "float16", "float32")
            projector_type: Projector architecture ("mlp", "mosa", "moe", "qformer")
            use_lora: Enable LoRA adapters for Stage 2 fine-tuning
            use_specaugment: Enable SpecAugment data augmentation
        """
        # Set default generation parameters (greedy decoding only)
        generation_defaults = {
            "num_beams": 1,
            "max_new_tokens": 128,
            "min_new_tokens": 0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,  # Prevent repeating 3-grams like "so so so"
            "use_cache": True,
        }

        # Apply defaults (config.json values take precedence)
        kwargs = {**generation_defaults, **kwargs}

        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id
        self.attn_implementation = attn_implementation
        self.model_dtype = model_dtype
        self.system_prompt = system_prompt
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        # Default conv layers for Whisper/GLM-ASR: [(pad, kernel, stride), ...]
        self.encoder_conv_layers = encoder_conv_layers or [(1, 3, 1), (1, 3, 2)]
        self.audio_sample_rate = audio_sample_rate
        self.projector_init_std = projector_init_std
        self.projector_pool_stride = projector_pool_stride
        self.downsample_rate = downsample_rate
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_type = projector_type
        self.projector_num_layers = projector_num_layers
        self.projector_dropout = projector_dropout
        # MoE-specific configuration
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef
        # QFormer-specific configuration
        self.qformer_window_size = qformer_window_size
        self.qformer_hidden_size = qformer_hidden_size
        self.qformer_num_layers = qformer_num_layers
        self.qformer_num_heads = qformer_num_heads
        self.qformer_intermediate_size = qformer_intermediate_size
        self.label_smoothing = label_smoothing
        self.inference_warmup_tokens = inference_warmup_tokens
        # SpecAugment configuration
        self.use_specaugment = use_specaugment
        self.num_time_masks = num_time_masks
        self.time_mask_length = time_mask_length
        self.num_freq_masks = num_freq_masks
        self.freq_mask_length = freq_mask_length
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

        # Generation parameters (use explicit value if provided, else use default)
        self.num_beams = num_beams if num_beams is not None else generation_defaults["num_beams"]
        self.max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else generation_defaults["max_new_tokens"]
        )
        self.min_new_tokens = (
            min_new_tokens if min_new_tokens is not None else generation_defaults["min_new_tokens"]
        )
        self.repetition_penalty = (
            repetition_penalty
            if repetition_penalty is not None
            else generation_defaults["repetition_penalty"]
        )
        self.length_penalty = (
            length_penalty if length_penalty is not None else generation_defaults["length_penalty"]
        )
        self.no_repeat_ngram_size = (
            no_repeat_ngram_size
            if no_repeat_ngram_size is not None
            else generation_defaults["no_repeat_ngram_size"]
        )
        self.use_cache = use_cache if use_cache is not None else generation_defaults["use_cache"]

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
