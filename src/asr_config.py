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
        num_beams: int = 5,
        system_prompt: str = "/no_think You are a helpful assistant.",
        encoder_dim: int = None,
        llm_dim: int = None,
        **kwargs,
    ):
        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id
        self.attn_implementation = attn_implementation
        self.model_dtype = model_dtype
        self.audio_downsample_rate = audio_downsample_rate
        self.num_beams = num_beams
        self.system_prompt = system_prompt
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
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
