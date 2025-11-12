import numpy as np
import pytest
from hydra import compose, initialize
from transformers import AutoTokenizer, WhisperFeatureExtractor

from src.train import DataCollator, DatasetLoader


def test_multi_task_dataset_loading():
    """
    Tests that the DatasetLoader correctly loads and interleaves multiple datasets
    when using the multi_task_complete configuration.
    """
    # Initialize Hydra and compose the configuration
    with initialize(version_base="1.1", config_path="../configs/hydra", job_name="test_app"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=multi_task_complete",
                "data.dataset_cache_dir=/tmp/tiny_audio_test_cache",
            ],
        )

        # Instantiate the DatasetLoader with the loaded configuration
        dataset_loader = DatasetLoader(cfg)

        # Load the training dataset
        train_ds, val_ds = dataset_loader.load()

        # Check that train_ds is not None
        assert train_ds is not None
        # In this case, val_ds can be None, so we don't assert on it.

        # Take a sample from the interleaved dataset
        num_samples_to_check = 1000
        samples = list(train_ds.take(num_samples_to_check))

        # Check that we got the number of samples we asked for
        assert len(samples) == num_samples_to_check

        # --- Start of new code for printing samples ---
        print("\n--- First 10 samples with prompts ---")

        # We need a collator to generate the prompts
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.decoder_model_name, trust_remote_code=True
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        data_collator = DataCollator(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            sample_rate=16000,
        )

        for i, sample in enumerate(samples[:10]):
            print(f"\n--- Sample {i + 1} ---")
            print(f"Task: {sample['task']}")
            print(f"Text: {sample['text']}")

            # Generate the prompt
            batch = data_collator([sample])
            decoded_text = tokenizer.decode(batch["input_ids"][0])

            # Extract just the prompt part
            prompt = decoded_text.split("assistant\n")[0] + "assistant\n"
            print(f"Prompt: {prompt}")
        print("\n" + "=" * 40 + "\n")
        # --- End of new code ---

        # Collect the tasks from the samples
        tasks = {sample["task"] for sample in samples}

        # Define the expected tasks
        expected_tasks = {"transcribe", "describe", "emotion"}

        # Assert that all expected tasks are present in the sample
        assert tasks == expected_tasks


@pytest.mark.parametrize(
    "task, expected_instruction",
    [
        ("transcribe", "Transcribe: <audio>"),
        ("describe", "Describe: <audio>"),
        ("emotion", "Emotion: <audio>"),
    ],
)
def test_prompt_strategy(task, expected_instruction, monkeypatch):
    """
    Tests that the DataCollator uses the correct prompt instruction for each task.
    """
    # Mock the _extract_audio method to avoid dealing with real audio objects
    monkeypatch.setattr(DataCollator, "_extract_audio", lambda self, audio: np.zeros(16000))

    with initialize(version_base="1.1", config_path="../configs/hydra", job_name="test_app"):
        cfg = compose(config_name="config", overrides=["data=multi_task_complete"])

        # Load the correct tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.decoder_model_name, trust_remote_code=True
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

        # Instantiate the DataCollator
        data_collator = DataCollator(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            sample_rate=16000,
        )

        # Create a dummy sample
        dummy_sample = {
            "audio": {"array": np.zeros(16000), "sampling_rate": 16000},
            "text": "dummy text",
            "task": task,
        }

        # Collate the sample
        batch = data_collator([dummy_sample])

        # Decode the input_ids
        decoded_text = tokenizer.decode(batch["input_ids"][0])

        # Assert that the expected instruction is in the decoded text
        assert expected_instruction in decoded_text
