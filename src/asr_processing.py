import json
from pathlib import Path

import transformers
from transformers import AutoTokenizer, ProcessorMixin

from .asr_config import ASRConfig


class ASRProcessor(ProcessorMixin):
    """Generic processor that can handle both Wav2Vec2 and Whisper feature extractors."""

    feature_extractor_class: str = "AutoFeatureExtractor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoFeatureExtractor

        # Load feature extractor and tokenizer from saved model directory
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True, **kwargs
        )

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def save_pretrained(self, save_directory, **kwargs):
        """Override save_pretrained to avoid attribute errors from base class."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the feature extractor (this creates preprocessor_config.json with all feature extractor settings)
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory)

        # Save the tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

        # Load the existing preprocessor_config.json and add processor-specific metadata
        config_path = save_path / "preprocessor_config.json"
        if config_path.exists():
            with config_path.open() as f:
                processor_config = json.load(f)
        else:
            processor_config = {}

        # Add/update processor metadata while preserving feature extractor settings
        feature_extractor_type = self.feature_extractor.__class__.__name__
        processor_config.update(
            {
                "processor_class": self.__class__.__name__,
                "feature_extractor_class": self.feature_extractor_class,
                "tokenizer_class": self.tokenizer_class,
                "feature_extractor_type": feature_extractor_type,  # Dynamic based on actual type
                "auto_map": {"AutoProcessor": "asr_processing.ASRProcessor"},
            }
        )

        # Save the merged config
        with config_path.open("w") as f:
            json.dump(processor_config, f, indent=2)


ASRProcessor.register_for_auto_class()
transformers.AutoProcessor.register(ASRConfig, ASRProcessor)
