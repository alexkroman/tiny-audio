import transformers
from transformers import AutoFeatureExtractor, AutoTokenizer, ProcessorMixin

# Handle both package and standalone imports
try:
    from .asr_config import ASRConfig
except ImportError:
    from asr_config import ASRConfig  # type: ignore[no-redef]


class ASRProcessor(ProcessorMixin):
    """Generic processor that can handle both Wav2Vec2 and Whisper feature extractors."""

    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = ASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        feature_extractor = AutoFeatureExtractor.from_pretrained(config.audio_model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True, **kwargs
        )

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)


ASRProcessor.register_for_auto_class()
transformers.AutoProcessor.register(ASRConfig, ASRProcessor)
