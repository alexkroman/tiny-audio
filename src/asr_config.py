import transformers


class ASRConfig(transformers.PretrainedConfig):
    model_type = "asr_model"
    is_composition = True

    def __init__(
        self,
        audio_model_id: str = "facebook/hubert-large-ls960-ft",
        text_model_id: str = "Qwen/Qwen3-8B",
        attn_implementation: str = "sdpa",
        model_dtype: str = "bfloat16",
        audio_downsample_rate: int = 5,
        num_beams: int = 1,
        system_prompt: str = "/no_think /system_override",
        user_prompt: str = "Transcribe in English: <audio>",
        encoder_dim: int = None,
        llm_dim: int = None,
        projector_hidden_dim: int = 8192,
        # Audio processing constants
        audio_sample_rate: int = 16000,
        # Projector initialization constants
        projector_init_std: float = 0.02,
        # LoRA default parameters
        lora_default_dropout: float = 0.0,
        # Inference parameters
        inference_diversity_penalty: float = 0.5,
        inference_warmup_tokens: int = 10,
        # Generation parameters
        max_new_tokens: int = 256,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        top_k: int = 0,
        top_p: float = 0.8,
        use_cache: bool = True,
        **kwargs,
    ):
        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id
        self.attn_implementation = attn_implementation
        self.model_dtype = model_dtype
        self.audio_downsample_rate = audio_downsample_rate
        self.num_beams = num_beams
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.projector_hidden_dim = projector_hidden_dim
        self.audio_sample_rate = audio_sample_rate
        self.projector_init_std = projector_init_std
        self.lora_default_dropout = lora_default_dropout
        self.inference_diversity_penalty = inference_diversity_penalty
        self.inference_warmup_tokens = inference_warmup_tokens
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.use_cache = use_cache
        if "audio_config" not in kwargs:
            self.audio_config = transformers.AutoConfig.from_pretrained(audio_model_id)
        else:
            self.audio_config = kwargs.pop("audio_config")

        if "text_config" not in kwargs:
            self.text_config = transformers.AutoConfig.from_pretrained(
                text_model_id, trust_remote_code=True
            )
        else:
            self.text_config = kwargs.pop("text_config")

        # Ensure configs are PretrainedConfig objects (in case loaded from dict)
        if isinstance(self.text_config, dict):
            # Reconstruct config from dict using the model_type stored in the dict
            model_type = self.text_config.get("model_type")
            if model_type:
                config_class = transformers.AutoConfig.for_model(model_type).__class__
                self.text_config = config_class(**self.text_config)
            else:
                # Fallback: try to load from model_id
                self.text_config = transformers.AutoConfig.from_pretrained(
                    text_model_id, trust_remote_code=True
                )

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

    def to_dict(self):
        """Override to ensure all critical fields are serialized."""
        output = super().to_dict()

        # Explicitly ensure these fields are saved
        output["encoder_dim"] = self.encoder_dim
        output["llm_dim"] = self.llm_dim
        output["projector_hidden_dim"] = self.projector_hidden_dim
        output["audio_downsample_rate"] = self.audio_downsample_rate
        output["system_prompt"] = self.system_prompt
        output["user_prompt"] = self.user_prompt
        output["num_beams"] = self.num_beams
        output["audio_sample_rate"] = self.audio_sample_rate
        output["projector_init_std"] = self.projector_init_std
        output["lora_default_dropout"] = self.lora_default_dropout
        output["inference_diversity_penalty"] = self.inference_diversity_penalty
        output["inference_warmup_tokens"] = self.inference_warmup_tokens
        output["max_new_tokens"] = self.max_new_tokens
        output["min_new_tokens"] = self.min_new_tokens
        output["do_sample"] = self.do_sample
        output["top_k"] = self.top_k
        output["top_p"] = self.top_p
        output["use_cache"] = self.use_cache

        return output
