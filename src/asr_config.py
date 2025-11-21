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
        projector_hidden_dim: Optional[int] = None,
        projector_dropout: float = 0.05,  # Dropout rate for projector layers
        projector_input_noise: float = 0.02,  # Input noise for projector
        inference_diversity_penalty: float = 0.0,
        inference_warmup_tokens: int = 10,
        use_4bit_quantization: bool = True,  # Enable 4-bit quantization
        bnb_4bit_compute_dtype: str = "bfloat16",  # Compute dtype for 4-bit
        bnb_4bit_quant_type: str = "nf4",  # Quantization type: nf4 or fp4
        bnb_4bit_use_double_quant: bool = True,  # Use double quantization
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
        # Set default generation parameters
        generation_defaults = {
            "num_beams": 1,
            "max_new_tokens": 128,
            "min_new_tokens": 1,
            "do_sample": False,
            "repetition_penalty": 1.05,
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
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_dropout = projector_dropout
        self.projector_input_noise = projector_input_noise
        self.inference_diversity_penalty = inference_diversity_penalty
        self.inference_warmup_tokens = inference_warmup_tokens
        self.use_4bit_quantization = use_4bit_quantization
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
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


transformers.AutoConfig.register("asr_model", ASRConfig)
