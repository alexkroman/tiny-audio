# Quick Start Guide

Get up and running with the Tiny Audio course in 5 minutes!

## Prerequisites

- Python 3.10 or newer
- Git
- 8GB RAM minimum
- 20GB free disk space

## Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio

# Install Poetry (if you don't have it)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

**Expected time**: 5-10 minutes (downloads PyTorch, Transformers, etc.)

## Step 2: Download Sample Audio

```bash
# Download sample audio files for exercises
poetry run download-samples
```

**Expected time**: 2-3 minutes (downloads 346MB of LibriSpeech samples)

This creates:
- `samples/LibriSpeech/test-clean/` - Various audio samples
- `test.wav` - Default test file (symlinked to first sample)

## Step 3: Verify Your Setup

Run the verification script to check everything is working:

```bash
poetry run verify-setup
```

This checks:
- Python version (3.10+)
- All required packages
- Sample audio files
- Model config loading

**Expected output**: All checks should pass ✓

## Step 4: Test Inference

Run a quick inference test:

```bash
poetry run python test_inference.py
```

**Expected output**:
```
Loading model...
✓ Model loaded!
Transcribing test.wav...

Transcription:
[Your audio transcription will appear here]
```

**Note**: First run downloads ~4GB of model weights from HuggingFace Hub. Be patient!

## Step 5: Start the Course!

You're ready to begin! Start with:

**[Class 1: Introduction and Setup](./course/1-introduction-and-setup.md)**

## Common Issues

### "ModuleNotFoundError: No module named 'librosa'"

**Fix**: Make sure you ran `poetry install` to install all dependencies.

### "FileNotFoundError: [Errno 2] No such file or directory: 'test.wav'"

**Fix**: Run `poetry run download-samples` to download sample audio files.

### Model download is very slow

**Fix**:
- The model is 4GB, first download takes time
- Ensure you have a stable internet connection
- Model is cached in `~/.cache/huggingface/` for future use

### Out of memory errors

**Fix**:
- Close other applications
- For visualization exercises (Class 2-3), try smaller audio files
- Consider using a machine with more RAM for training (Class 4+)

## What's Included

### Course Materials
- **docs/course/** - 6 classes covering ASR from scratch
- **docs/COURSE_ISSUES_AND_FIXES.md** - Known issues and solutions

### Exercise Scripts
All exercise scripts are pre-created in the root directory:

**Class 1**:
- `test_inference.py` - Run ASR inference
- `test_config.py` - Inspect model config
- `test_count_params.py` - Count parameters

**Class 2**:
- `explore_audio.py` - Visualize audio processing
- `explore_hubert.py` - Explore encoder outputs
- `count_params.py` - Count parameters by component

**Class 3**:
- `trace_projector.py` - Trace projector layers
- `visualize_projector.py` - Visualize embeddings
- `test_projector_config.py` - Compare configurations

## Next Steps

1. **Learn**: Work through the 6-class course
2. **Train**: Start your own training run (Class 4)
3. **Evaluate**: Measure your model's performance (Class 5)
4. **Deploy**: Push to HuggingFace Hub (Class 6)
5. **Contribute**: Add your results to the leaderboard!

## Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Join the community discussions
- **Documentation**: See README.md and course materials

## Course Structure

Each class is ~1 hour:
- **20 min**: Lecture (concepts and theory)
- **40 min**: Hands-on workshop (coding exercises)

**Total time commitment**: 6 hours
**Total cost for GPU training**: ~$12 (or free with local GPU)

---

**Ready to build your own speech recognition model?** 🚀

Start here: **[Class 1: Introduction and Setup](./course/1-introduction-and-setup.md)**
