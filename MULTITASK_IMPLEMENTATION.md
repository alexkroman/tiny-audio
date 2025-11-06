# Multi-Task Audio Model Implementation

## Overview

This implementation adds three distinct tasks to the audio model, all using the same underlying architecture but with task-specific prompts:

1. **Transcribe**: Speech-to-text transcription (default)
2. **Continue**: Dialogue continuation prediction
3. **Describe**: Audio description/captioning

## Tasks

### 1. Transcription Task
- **Prompt**: `"Transcribe: <audio>"`
- **Dataset**: LoquaciousSet (speechbrain/LoquaciousSet)
- **Purpose**: Convert speech audio to text
- **Config**: `configs/hydra/data/loquacious_clean.yaml`

### 2. Continuation Task
- **Prompt**: `"Continue: <audio>"`
- **Dataset**: SODA-Audio (fixie-ai/soda-audio)
- **Purpose**: Predict the next turn in a dialogue given the previous audio
- **Config**: `configs/hydra/data/soda_continue.yaml`
- **Input**: `audio_second_last_turn` column
- **Target**: `alt_last_turn` column

### 3. Description Task
- **Prompt**: `"Describe: <audio>"`
- **Dataset**: AudioSet Strong (CLAPv2/audioset_strong)
- **Purpose**: Generate descriptions of sounds in audio
- **Config**: `configs/hydra/data/audioset_describe.yaml`

## Configuration Files

### Single-Task Training
```bash
# Transcription only
poetry run python src/train.py data=loquacious_clean

# Continuation only
poetry run python src/train.py +experiments=continue_test

# Description only
poetry run python src/train.py +experiments=describe_test
```

### Multi-Task Training
```bash
# Two tasks (transcription + continuation)
poetry run python src/train.py +experiments=multi_task_test

# All three tasks
poetry run python src/train.py +experiments=multi_task_all_test
```

## Data Configuration

### Multi-Task Configuration Example
```yaml
# configs/hydra/data/multi_task_all.yaml
datasets:
  - path: speechbrain/LoquaciousSet
    name: clean
    task: transcribe
    sampling_weight: 0.4  # 40% of samples

  - path: fixie-ai/soda-audio
    name: default
    task: continue
    sampling_weight: 0.3  # 30% of samples

  - path: CLAPv2/audioset_strong
    task: describe
    sampling_weight: 0.3  # 30% of samples
```

## Implementation Details

### Training Code Updates

1. **DataLoader** (`src/train.py`):
   - Added task field to each dataset sample
   - Supports sampling weights for balanced multi-task training
   - Handles datasets with/without configuration names

2. **DataCollator** (`src/train.py`):
   - Dynamically selects prompt based on task type
   - Maintains same label masking strategy for all tasks

### Inference Updates

1. **ASR Pipeline** (`src/asr_pipeline.py`):
   - Added `task` parameter to pipeline call
   - Passes task through to model's generate method

2. **ASR Model** (`src/asr_modeling.py`):
   - Generate method accepts `task` parameter
   - Automatically sets appropriate prompt based on task

## Usage Examples

### Training
```python
# Single task (default transcription)
poetry run python src/train.py

# Multi-task with custom weights
poetry run python src/train.py data=multi_task_all
```

### Inference
```python
from src.asr_pipeline import ASRPipeline
from src.asr_modeling import ASRModel

# Load trained model
model = ASRModel.from_pretrained("path/to/model")
pipeline = ASRPipeline(model)

# Transcription (default)
result = pipeline(audio_file)
# or explicitly:
result = pipeline(audio_file, task="transcribe")

# Continuation
result = pipeline(audio_file, task="continue")

# Description
result = pipeline(audio_file, task="describe")
```

### Direct Model Usage
```python
# Process audio
input_values = model.feature_extractor(
    audio_array,
    sampling_rate=16000,
    return_tensors="pt"
).input_values

# Generate with specific task
output = model.generate(
    input_values,
    task="describe",  # or "transcribe", "continue"
    max_new_tokens=200
)

# Decode output
text = model.tokenizer.decode(output[0], skip_special_tokens=True)
```

## Key Features

- **Unified Architecture**: Same model handles all tasks
- **Task-Specific Prompts**: Different prompts guide the model's behavior
- **Weighted Sampling**: Control task distribution during multi-task training
- **Streaming Support**: All datasets use streaming for memory efficiency
- **Simple Integration**: Minimal code changes to add new tasks

## Adding New Tasks

To add a new task:

1. Create dataset config in `configs/hydra/data/`
2. Add task handling in DataCollator (`src/train.py`)
3. Add task handling in model generate (`src/asr_modeling.py`)
4. Create experiment config for testing

Example for a hypothetical "Translate" task:
```python
# In DataCollator
elif task == "translate":
    messages.append({
        "role": "user",
        "content": "Translate: <audio>"
    })

# In ASRModel.generate()
elif task == "translate":
    user_prompt = "Translate: <audio>"
```

## Performance Notes

- Training may be slow to start due to model size and dataset loading
- Use streaming mode to reduce memory usage
- Adjust sampling weights to balance task importance
- Consider using smaller batch sizes for memory-constrained systems