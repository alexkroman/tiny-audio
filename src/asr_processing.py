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

        # For Whisper models, ensure we load the feature extractor with correct mel bins
        is_whisper = "whisper" in config.audio_model_id.lower()
        if is_whisper:
            from transformers import WhisperConfig, WhisperFeatureExtractor

            encoder_config = WhisperConfig.from_pretrained(config.audio_model_id)
            num_mel_bins = encoder_config.num_mel_bins
            feature_extractor = WhisperFeatureExtractor.from_pretrained(
                config.audio_model_id,
                feature_size=num_mel_bins,  # Override to match encoder's mel bins (128 for V3 Turbo)
            )
        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained(config.audio_model_id)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True, **kwargs
        )

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def save_pretrained(self, save_directory, **kwargs):
        """Override save_pretrained to avoid attribute errors from base class."""
        import json
        from pathlib import Path

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
