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

    def save_pretrained(self, save_directory, **kwargs):
        """Override save_pretrained to avoid attribute errors from base class."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save the feature extractor
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory)

        # Save the tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

        # Save processor config
        # Detect the actual feature extractor type
        feature_extractor_type = self.feature_extractor.__class__.__name__

        processor_config = {
            "processor_class": self.__class__.__name__,
            "feature_extractor_class": self.feature_extractor_class,
            "tokenizer_class": self.tokenizer_class,
            "feature_extractor_type": feature_extractor_type,  # Dynamic based on actual type
            "auto_map": {
                "AutoProcessor": "asr_processing.ASRProcessor"
            }
        }

        with open(os.path.join(save_directory, "preprocessor_config.json"), "w") as f:
            json.dump(processor_config, f, indent=2)


ASRProcessor.register_for_auto_class()
transformers.AutoProcessor.register(ASRConfig, ASRProcessor)
