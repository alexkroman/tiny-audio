from typing import Optional

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
        audio_downsample_rate: int = 5,  # Deprecated: use projector_pool_stride instead
        num_beams: Optional[int] = None,
        system_prompt: str = "/no_think /system_override",
        user_prompt: str = "Transcribe: <audio>",
        encoder_dim: Optional[int] = None,
        llm_dim: Optional[int] = None,
        # Audio processing constants
        audio_sample_rate: int = 16000,
        # Projector initialization constants
        projector_init_std: float = 0.02,
        projector_dropout: float = 0.05,
        projector_pool_stride: int = 2,  # AvgPool1d stride (2 = 4x total with Whisper, 1 = no pooling)
        # LoRA default parameters
        lora_default_dropout: float = 0.0,
        # Inference parameters
        inference_diversity_penalty: float = 0.5,
        inference_warmup_tokens: int = 10,
        # Generation parameters
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        # Set default values for generation params if not present in kwargs
        # This allows config.json values to take precedence when loading from pretrained
        generation_defaults = {
            "num_beams": 1,
            "max_new_tokens": 128,
            "min_new_tokens": 1,
            "do_sample": False,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 0,
            "use_cache": True,  # Now enabled - we pre-expand audio tokens for consistent sequence length
        }

        # Only add sampling parameters if do_sample=True
        do_sample_value = kwargs.get("do_sample", generation_defaults["do_sample"])
        if do_sample_value:
            generation_defaults["temperature"] = 1.0
            generation_defaults["top_k"] = 0
            generation_defaults["top_p"] = 0.8

        # Only add early_stopping if using beam search
        num_beams_value = kwargs.get("num_beams", generation_defaults["num_beams"])
        if num_beams_value > 1:
            generation_defaults["early_stopping"] = True

        for param_name, default_value in generation_defaults.items():
            if param_name not in kwargs:
                kwargs[param_name] = default_value

        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id
        self.attn_implementation = attn_implementation
        self.model_dtype = model_dtype
        self.audio_downsample_rate = audio_downsample_rate
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.audio_sample_rate = audio_sample_rate
        self.projector_init_std = projector_init_std
        self.projector_dropout = projector_dropout
        self.projector_pool_stride = projector_pool_stride
        self.lora_default_dropout = lora_default_dropout
        self.inference_diversity_penalty = inference_diversity_penalty
        self.inference_warmup_tokens = inference_warmup_tokens
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
        output["audio_downsample_rate"] = self.audio_downsample_rate
        output["projector_pool_stride"] = self.projector_pool_stride
        output["system_prompt"] = self.system_prompt
        output["user_prompt"] = self.user_prompt
        output["num_beams"] = self.num_beams
        output["audio_sample_rate"] = self.audio_sample_rate
        output["projector_init_std"] = self.projector_init_std
        output["projector_dropout"] = self.projector_dropout
        output["lora_default_dropout"] = self.lora_default_dropout
        output["inference_diversity_penalty"] = self.inference_diversity_penalty
        output["inference_warmup_tokens"] = self.inference_warmup_tokens
        output["max_new_tokens"] = self.max_new_tokens
        output["min_new_tokens"] = self.min_new_tokens
        output["do_sample"] = self.do_sample
        output["temperature"] = self.temperature
        output["top_k"] = self.top_k
        output["top_p"] = self.top_p
        output["repetition_penalty"] = self.repetition_penalty
        output["length_penalty"] = self.length_penalty
        output["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        output["early_stopping"] = self.early_stopping
        output["use_cache"] = self.use_cache

        return output


# Register the config with transformers
# This is needed for AutoConfig.from_pretrained to work
transformers.AutoConfig.register("asr_model", ASRConfig)
