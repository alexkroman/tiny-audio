#!/bin/bash
set -e

# Run minimal training for e2e test
exec uv run src/train.py \
    +experiments=mac_minimal \
    training.max_steps=1 \
    training.save_steps=1 \
    training.eval_steps=1 \
    training.logging_steps=1 \
    data.max_train_samples=1 \
    training.gradient_checkpointing=false \
    +output_dir=outputs/test_e2e_model