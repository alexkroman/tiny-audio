# Tiny Audio development tasks
# Install just: brew install just (macOS) or cargo install just

# Default recipe - show available commands
default:
    @just --list

# Run ruff linter
lint:
    ruff check src scripts

# Format code with black, ruff, and mdformat
format:
    black src scripts
    ruff format src scripts
    ruff check --fix src scripts
    find . -name "*.md" -not -name "MODEL_CARD.md" -not -path "./.venv/*" -not -path "./docs/course/*" | xargs mdformat

# Run type checkers (mypy and pyright)
type-check:
    mypy src
    pyright src

# Run pytest tests
test:
    pytest -v

# Run all checks (lint + type-check)
check: lint type-check

# Run all checks and tests
all: check test

# Train with an experiment config
train experiment="mlp":
    python -m scripts.train +experiments={{experiment}}

# Evaluate a model
eval model dataset="loquacious":
    python -m scripts.eval.cli {{model}} --datasets {{dataset}}

# Deploy to RunPod
deploy host port:
    python -m scripts.runpod deploy {{host}} {{port}}

# Start remote training
remote-train host port experiment="mlp":
    python -m scripts.runpod train {{host}} {{port}} -e {{experiment}}

# Attach to remote session
remote-attach host port:
    python -m scripts.runpod attach {{host}} {{port}}

# Push model to HuggingFace Hub
push-hub:
    python -m scripts.push_to_hub
