import transformers
from transformers import AutoFeatureExtractor, AutoTokenizer, Wav2Vec2Processor

# Handle both package and standalone imports
try:
    from .asr_config import ASRConfig
except ImportError:
    from asr_config import ASRConfig  # type: ignore[no-redef]


class ASRProcessor(Wav2Vec2Processor):
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
