"""Speaker-attributed ASR: LoRA-tune Qwen3-ASR to diarize while it transcribes.

The model writes a multi-speaker window as `<SPK_1>text<SPK_2>text...`,
speakers numbered by first appearance, so one greedy decode answers who
spoke, when (in turn order) and what. Trained on AMI headset segments re-mixed
on their real meeting timeline; scored by cpWER.

    data.py     window planning/mixing, targets, dataset, cpWER metrics
    model.py    speaker tokens, LoRA, decode that keeps the speaker tokens
    config.py   Hydra config + pool signature (configs/speaker_asr/, presets in experiment/)
    train.py    Hydra entry point: `python -m scripts.speaker_asr.train +experiment=v1`
    cli.py      `ta speaker-asr build-pool | evaluate`
"""
