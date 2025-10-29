# Course Materials - Issues Found and Fixes Needed

## Summary

This document tracks issues found while going through the course materials as a student would, and what needs to be fixed.

## Global Issues

### 1. Missing Dependencies for Course Exercises

**Issue**: Classes 2 and 3 require `librosa` and `matplotlib` for visualization exercises, but these are not in `pyproject.toml`.

**Impact**: Students will get import errors when running exercises.

**Fix Needed**:
```bash
poetry add librosa matplotlib
```

Or add to pyproject.toml:
```toml
[tool.poetry.dependencies]
librosa = "^0.10.0"
matplotlib = "^3.7.0"
```

### 2. No Sample Audio Files

**Issue**: All exercises require audio files, but none are provided in the repository.

**Impact**: Students must find/record their own audio before starting any exercises.

**Fix Options**:
1. Add a few sample audio files to a `samples/` directory
2. Provide a script to download sample files from LibriSpeech
3. Add better instructions for generating test audio

**Recommendation**: Create `scripts/download_samples.py`:
```python
import urllib.request
from pathlib import Path

# Download a few small LibriSpeech samples
samples_dir = Path("samples")
samples_dir.mkdir(exist_ok=True)

urls = [
    "https://www.openslr.org/resources/12/... (specific file URLs)"
]

for url in urls:
    filename = samples_dir / url.split("/")[-1]
    if not filename.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
```

## Class 1: Introduction and Setup

### Issues Found:
1. ✅ Exercise scripts created (test_inference.py, test_config.py, test_count_params.py)
2. ⚠️ No audio files to test with
3. ⚠️ wget command for LibriSpeech might not work on Windows

### Fixes Needed:
- Add cross-platform audio download script
- Provide at least one sample audio file

## Class 2: Audio Processing and Encoders

### Issues Found:
1. ✅ Exercise scripts created (explore_audio.py, explore_hubert.py, count_params.py)
2. ❌ Missing dependencies: librosa, matplotlib
3. ⚠️ Instructions say "poetry add librosa matplotlib" but this isn't in the base install
4. ⚠️ No audio files to test with

### Fixes Needed:
- Add librosa and matplotlib to dependencies
- Add note that Step 1 requires adding these packages
- Consider making visualization optional or providing pre-rendered images

## Class 3: Language Models and Projectors

### Issues Found:
1. ✅ Exercise scripts created (trace_projector.py, visualize_projector.py, test_projector_config.py)
2. ❌ Missing dependencies: librosa, matplotlib (for visualize_projector.py)
3. ⚠️ No audio files to test with
4. ⚠️ test_projector_config.py uses SimpleNamespace but doesn't need it (can remove import)

### Fixes Needed:
- Same dependency issues as Class 2
- Clean up unused import

## Class 4: Training

### Needs Review:
- Check if training commands work as documented
- Verify config file paths are correct
- Test that students can actually start a training run
- Check GPU setup instructions are accurate

### Potential Issues:
- Cloud GPU setup might need more detailed instructions
- Students may not have access to RunPod/vast.ai accounts
- Local training on Apple Silicon needs special handling

## Class 5: Evaluation and Debugging

### Needs Review:
- Check if evaluation scripts work
- Verify dataset paths are correct
- Test that WER calculations are accurate

## Class 6: Publishing and Deployment

### Needs Review:
- Verify HuggingFace Hub push works
- Check model card template
- Test leaderboard submission process

## Scripts Created (Ready to Use)

All these scripts have been created in the root directory:

### Class 1:
- `test_inference.py` - Basic ASR inference
- `test_config.py` - Inspect model configuration
- `test_count_params.py` - Count model parameters

### Class 2:
- `explore_audio.py` - Visualize audio processing
- `explore_hubert.py` - Explore HuBERT encoder outputs
- `count_params.py` - Count parameters by component

### Class 3:
- `trace_projector.py` - Trace through projector layers
- `visualize_projector.py` - Visualize embedding distributions
- `test_projector_config.py` - Compare projector configurations

## Recommended Actions

### High Priority:
1. ✅ Add librosa and matplotlib to dependencies
2. ✅ Provide sample audio files or download script
3. ⬜ Test all exercises end-to-end with the created scripts
4. ⬜ Update course materials with actual output from running scripts

### Medium Priority:
1. ⬜ Add troubleshooting section for common errors
2. ⬜ Create a "Quick Start" that downloads everything students need
3. ⬜ Add estimated time for model downloads (4GB for tiny-audio, etc.)

### Low Priority:
1. ⬜ Add video walkthroughs
2. ⬜ Create Jupyter notebook versions of exercises
3. ⬜ Add solutions to exercises

## Testing Checklist

- [ ] Install fresh environment and run through Class 1
- [ ] Verify all scripts can import required modules
- [ ] Test with sample audio file
- [ ] Run through Class 2 visualization exercises
- [ ] Run through Class 3 projector exercises
- [ ] Start a training run (even just a few steps)
- [ ] Test evaluation pipeline
- [ ] Test model push to Hub

## Notes

- All scripts reference "test.wav" as the audio file - students need to update this path
- Scripts assume the model "mazesmazes/tiny-audio" exists on HuggingFace Hub
- Some scripts may require significant download time on first run
