from typing import Optional

import transformers


class ASRConfig(transformers.PretrainedConfig):
    """Configuration class for the ASR model."""

    model_type = "asr_model"
    is_composition = True

    # Generation defaults
    GENERATION_DEFAULTS = {
        "num_beams": 1,
        "max_new_tokens": 128,
        "min_new_tokens": 0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "use_cache": True,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }

    def __init__(
        self,
        # Model IDs
        audio_model_id: str = "zai-org/GLM-ASR-Nano-2512",
        text_model_id: str = "Qwen/Qwen3-0.6B",
        # Model settings
        attn_implementation: str = "sdpa",
        model_dtype: str = "bfloat16",
        system_prompt: str = "You are a helpful assistant.",
        enable_thinking: bool = False,
        # Encoder settings (auto-detected if None)
        encoder_dim: Optional[int] = None,
        llm_dim: Optional[int] = None,
        encoder_conv_layers: Optional[list] = None,
        audio_sample_rate: int = 16000,
        # Projector settings
        projector_type: str = "mlp",
        projector_pool_stride: int = 4,
        projector_hidden_dim: Optional[int] = None,
        projector_num_layers: int = 2,
        projector_init_std: float = 0.02,
        projector_dropout: float = 0.0,
        # MoE projector settings
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        router_aux_loss_coef: float = 0.01,
        # QFormer projector settings
        qformer_window_size: int = 15,
        qformer_hidden_size: Optional[int] = None,
        qformer_num_layers: int = 2,
        qformer_num_heads: int = 16,
        qformer_intermediate_size: Optional[int] = None,
        downsample_rate: int = 5,
        # Training settings (not saved to config.json for inference)
        use_specaugment: bool = False,
        num_time_masks: int = 2,
        time_mask_length: int = 10,
        num_freq_masks: int = 0,
        freq_mask_length: int = 10,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        freeze_projector: bool = False,
        label_smoothing: float = 0.0,
        # Audio Head / Depformer settings (Moshi-style depth transformer)
        use_audio_head: bool = False,
        audio_head_hidden_dim: int = 512,  # Legacy (use depformer_dim instead)
        depformer_dim: int = 1024,  # Depformer hidden dimension
        depformer_num_layers: int = 6,  # Depformer transformer layers
        codebook_size: int = 2048,  # Mimi codec vocabulary size
        num_codebooks: int = 8,  # Number of codebooks to predict
        freeze_audio_head: bool = False,  # Freeze entire audio head
        first_codebook_weight: float = 100.0,  # Weight for first codebook loss (semantic tokens)
        acoustic_delay: int = 1,  # Delay τ for acoustic codebooks (Moshi uses 1 or 2)
        **kwargs,
    ):
        # Merge generation defaults with kwargs (kwargs takes precedence)
        for key, default in self.GENERATION_DEFAULTS.items():
            if key not in kwargs:
                kwargs[key] = default

        # Core model settings
        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id
        self.attn_implementation = attn_implementation
        self.model_dtype = model_dtype
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking

        # Encoder settings
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.encoder_conv_layers = encoder_conv_layers or [(1, 3, 1), (1, 3, 2)]
        self.audio_sample_rate = audio_sample_rate

        # Projector settings
        self.projector_type = projector_type
        self.projector_pool_stride = projector_pool_stride
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_num_layers = projector_num_layers
        self.projector_init_std = projector_init_std
        self.projector_dropout = projector_dropout

        # MoE settings
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef

        # QFormer settings
        self.qformer_window_size = qformer_window_size
        self.qformer_hidden_size = qformer_hidden_size
        self.qformer_num_layers = qformer_num_layers
        self.qformer_num_heads = qformer_num_heads
        self.qformer_intermediate_size = qformer_intermediate_size
        self.downsample_rate = downsample_rate

        # Training settings
        self.use_specaugment = use_specaugment
        self.num_time_masks = num_time_masks
        self.time_mask_length = time_mask_length
        self.num_freq_masks = num_freq_masks
        self.freq_mask_length = freq_mask_length
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
        self.label_smoothing = label_smoothing

        # Audio Head / Depformer settings (Moshi-style depth transformer)
        self.use_audio_head = use_audio_head
        self.audio_head_hidden_dim = audio_head_hidden_dim  # Legacy
        self.depformer_dim = depformer_dim
        self.depformer_num_layers = depformer_num_layers
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.freeze_audio_head = freeze_audio_head
        self.first_codebook_weight = first_codebook_weight
        self.acoustic_delay = acoustic_delay

        # Generation parameters (from kwargs after merge with defaults)
        self.num_beams = kwargs.pop("num_beams")
        self.max_new_tokens = kwargs.pop("max_new_tokens")
        self.min_new_tokens = kwargs.pop("min_new_tokens")
        self.repetition_penalty = kwargs.pop("repetition_penalty")
        self.length_penalty = kwargs.pop("length_penalty")
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size")
        self.use_cache = kwargs.pop("use_cache")
        self.do_sample = kwargs.pop("do_sample")
        self.temperature = kwargs.pop("temperature")
        self.top_p = kwargs.pop("top_p")
        self.top_k = kwargs.pop("top_k")

        # Load sub-configs
        self.audio_config = kwargs.pop("audio_config", None)
        if self.audio_config is None:
            self.audio_config = transformers.AutoConfig.from_pretrained(
                audio_model_id, trust_remote_code=True
            )
            self.audio_config.dtype = model_dtype
        elif isinstance(self.audio_config, dict) and self.audio_config.get("model_type"):
            config_class = transformers.AutoConfig.for_model(
                self.audio_config["model_type"]
            ).__class__
            self.audio_config = config_class(**self.audio_config)

        self.text_config = kwargs.pop("text_config", None)
        if self.text_config is None:
            self.text_config = transformers.AutoConfig.from_pretrained(
                text_model_id, trust_remote_code=True
            )
            self.text_config.dtype = model_dtype
        elif isinstance(self.text_config, dict):
            config_class = transformers.AutoConfig.for_model(
                self.text_config["model_type"]
            ).__class__
            self.text_config = config_class(**self.text_config)

        super().__init__(**kwargs)

        # Pipeline configuration
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
