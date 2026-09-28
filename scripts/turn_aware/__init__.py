"""Turn-aware ASR: teach Qwen3-ASR to emit an end-of-turn token.

The model transcribes as usual and appends `<END_OF_TURN>` when, given only
the audio heard so far, the speaker has finished AND enough trailing silence
has been observed. Holding the turn is simply the absence of the token, so
transcription and endpointing come out of one greedy decode.

    data.py     pool construction, collator, metrics, predict_rows
    model.py    training-only model setup: marker-row init, LoRA
    config.py   Hydra config + pool signatures (configs/turn_aware/, presets in experiment/)
    train.py    Hydra entry point: `python -m scripts.turn_aware.train +experiment=v2`
    cli.py      `ta turn-aware build-pool | mine-pauses | evaluate | replay | set-threshold`
"""
