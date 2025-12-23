from typing import Optional

import transformers


class ASRConfig(transformers.PretrainedConfig):
    model_type = "asr_model"
    is_composition = True

    def __init__(
        self,
        audio_model_id: str = "openai/whisper-large-v3-turbo",
        text_model_id: str = "HuggingFaceTB/SmolLM3-3B",
        attn_implementation: str = "flash_attention_2",
        model_dtype: str = "bfloat16",
        num_beams: Optional[int] = None,
        system_prompt: str = "/no_think /system_override",
        user_prompt: str = "Transcribe: <audio>",
        encoder_dim: Optional[int] = None,
        llm_dim: Optional[int] = None,
        audio_sample_rate: int = 16000,
        projector_init_std: float = 0.02,
        projector_pool_stride: int = 2,
        downsample_rate: int = 16,
        projector_hidden_dim: Optional[int] = None,
        projector_type: str = "moe",  # "moe", "swiglu", "residual", "shared_moe", "mlp", "qformer"
        projector_num_layers: int = 2,  # Number of layers (for residual projector)
        projector_dropout: float = 0.05,  # Dropout rate for projector layers
        projector_input_noise: float = 0.02,  # Input noise for projector
        # MoE-specific configuration
        num_experts: int = 4,  # Number of experts in MoE projectors
        num_experts_per_tok: int = 2,  # Top-k experts per token
        router_aux_loss_coef: float = 0.01,  # Auxiliary loss coefficient for load balancing
        use_specaugment: bool = True,  # Apply SpecAugment during training
        # QFormer-specific configuration
        qformer_window_size: int = 100,  # Window size for QFormer processing
        qformer_hidden_size: Optional[int] = None,  # QFormer hidden size (defaults to encoder_dim)
        qformer_num_layers: int = 2,  # Number of QFormer transformer layers
        qformer_num_heads: int = 8,  # Number of attention heads in QFormer (must divide hidden size)
        qformer_intermediate_size: Optional[int] = None,  # FFN size (defaults to 4x hidden)
        label_smoothing: float = 0.0,  # Label smoothing for cross-entropy loss
        inference_diversity_penalty: float = 0.0,
        inference_warmup_tokens: int = 10,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        # Set default generation parameters (greedy decoding only)
        generation_defaults = {
            "num_beams": 1,
            "max_new_tokens": 96,
            "min_new_tokens": 0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "use_cache": True,
        }

        # Apply defaults (config.json values take precedence)
        kwargs = {**generation_defaults, **kwargs}

        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id
        self.attn_implementation = attn_implementation
        self.model_dtype = model_dtype
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.audio_sample_rate = audio_sample_rate
        self.projector_init_std = projector_init_std
        self.projector_pool_stride = projector_pool_stride
        self.downsample_rate = downsample_rate
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_type = projector_type
        self.projector_num_layers = projector_num_layers
        self.projector_dropout = projector_dropout
        self.projector_input_noise = projector_input_noise
        # MoE-specific configuration
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef
        self.use_specaugment = use_specaugment
        # QFormer-specific configuration
        self.qformer_window_size = qformer_window_size
        self.qformer_hidden_size = qformer_hidden_size
        self.qformer_num_layers = qformer_num_layers
        self.qformer_num_heads = qformer_num_heads
        self.qformer_intermediate_size = qformer_intermediate_size
        self.label_smoothing = label_smoothing
        self.inference_diversity_penalty = inference_diversity_penalty
        self.inference_warmup_tokens = inference_warmup_tokens

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
