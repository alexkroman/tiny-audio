"""Shell scripts run inside tmux on a RunPod pod for training and evaluation.

Pure string builders: they take the run parameters and return the bash text
`ta runpod train` / `ta runpod eval` upload and execute.
"""

from __future__ import annotations

# Repairs LD_LIBRARY_PATH so a --user torch can find the image's nvidia-* libs.
# Shared by every remote script, including the dependency installer in runpod.py.
NVIDIA_LD_PATH_FIX = """# RunPod images ship torch in system dist-packages alongside its nvidia-*
# CUDA wheels. When the project pins a different torch version it installs
# into --user and shadows the image copy, but the nvidia libs stay in the
# system tree -- so the loader cannot find e.g. libcusparseLt.so.0 and every
# `import torch` dies with ImportError. Observed on
# runpod/pytorch:...-torch291 against this repo's torch ~2.8.0 pin; it also
# broke the flash-attn build, whose metadata hook imports torch.
NVLIBS="$(python3 -c 'import glob;print(":".join(sorted(\
glob.glob("/usr/local/lib/python*/dist-packages/nvidia/*/lib"))))')"
# Spelled out with if/else on purpose: these scripts are built with Python
# f-strings, so shell brace-expansion syntax would be parsed as an f-string
# replacement field and raise NameError at build time.
if [ -n "$LD_LIBRARY_PATH" ]; then
  export LD_LIBRARY_PATH="$NVLIBS:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$NVLIBS"
fi
"""

# Every remote script shares this header: the fd limit, the nvidia-lib
# LD_LIBRARY_PATH repair, and the HF cache/token exports. It lived inline in all
# three builders below and had already drifted between them.
_SCRIPT_PREAMBLE = (
    """#!/bin/bash
# NOTE: "set -e" intentionally removed so session stays active on crash for debugging

ulimit -n 65536
{pip_install}export PATH="/root/.local/bin:$PATH"

"""
    + NVIDIA_LD_PATH_FIX
    + """export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/datasets
export HF_XET_HIGH_PERFORMANCE=1
export HF_TOKEN="{hf_token}"
# TileLang JIT-compiles fla's gated delta-rule kernels on first use (~8s each,
# a handful of them -- sequence length is marked dynamic in the kernel, so this
# is bounded warmup rather than per-step recompilation). Its cache defaults to
# ~/.tilelang/cache, i.e. /root, which is ephemeral container storage: every
# fresh pod would recompile from scratch. Point it at the persistent volume for
# the same reason HF_HOME is redirected above.
export TILELANG_CACHE_DIR=/workspace/.cache/tilelang
"""
)


def script_preamble(hf_token: str, *, pip_packages: str = "", extras: str = "") -> str:
    """Shared shell header for the remote train/eval scripts.

    Args:
        hf_token: Value exported as HF_TOKEN.
        pip_packages: Extra packages to install before the run; the pip line is
            omitted entirely when empty. Only the eval script needs one
            (modelscope) now that Xet has replaced hf_transfer.
        extras: Extra `export` lines appended to the header.
    """
    pip_install = (
        f"pip install {pip_packages} --quiet --root-user-action=ignore\n" if pip_packages else ""
    )
    preamble = _SCRIPT_PREAMBLE.format(pip_install=pip_install, hf_token=hf_token)
    return preamble + extras


def script_epilogue(label: str, finished: str) -> str:
    """Shared tail: report the exit code, then idle so tmux stays inspectable.

    Args:
        label: Name used in the success/failure banners.
        finished: Name used in the closing message.
    """
    return f"""
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "===== {label} Completed Successfully ====="
else
    echo "===== {label} Failed with exit code: $EXIT_CODE ====="
fi

echo "{finished} finished. Session will remain active for inspection."
sleep infinity
"""


def training_exports(wandb_run_id: str | None, wandb_resume: str | None) -> str:
    """The `export` block every remote training script runs under."""
    wandb_exports = ""
    if wandb_run_id:
        wandb_exports += f'export WANDB_RUN_ID="{wandb_run_id}"\n'
    if wandb_resume:
        wandb_exports += f'export WANDB_RESUME="{wandb_resume}"\n'

    return (
        "export TOKENIZERS_PARALLELISM=false\n"
        'export HF_DATASETS_AUDIO_DECODER="soundfile"\n'
        f"{wandb_exports}"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        "export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1\n"
        "export TORCH_CUDNN_BENCHMARK=1\n"
        # Keep the inductor + triton caches on local NVMe (/root/.cache/...), not
        # on the NFS-backed /workspace volume. /workspace previously caused ESTALE
        # (Errno 116, "Stale file handle") crashes inside Inductor's compile-worker
        # pool when the underlying NFS handle expired mid-write -- typical for any
        # parallel-write workload on a networked FS. The cost of putting these on
        # local NVMe is one cold-cache compile per pod boot (seconds-minutes);
        # the cost of ESTALE is a dead training job.
        "export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch_inductor\n"
        "export TRITON_CACHE_DIR=/root/.cache/triton\n"
        "export TORCHINDUCTOR_FX_GRAPH_CACHE=1\n"
        "export TORCH_DYNAMO_ALLOW_UNSPEC_INT_ON_NN_MODULE=1\n"
        "export TORCH_CUDA_GRAPHS_ENABLED=0\n"
    )


def build_training_script(
    experiment: str,
    hf_token: str,
    wandb_run_id: str | None,
    wandb_resume: str | None,
    extra_args: list[str],
) -> str:
    """Generate the training script content."""
    extra_args_str = " ".join(extra_args) if extra_args else ""
    extra_exports = training_exports(wandb_run_id, wandb_resume)
    body = f"""
cd /workspace
python -m scripts.train +experiments={experiment} {extra_args_str}"""
    return (
        script_preamble(hf_token, extras=extra_exports)
        + body
        + script_epilogue("Training", "Training script")
    )


def build_eval_script(
    hf_token: str,
    model: str,
    datasets: list[str],
    max_samples: int | None,
    assemblyai_api_key: str | None,
    assemblyai_model: str,
    num_workers: int,
    streaming: bool,
    extra_args: list[str] | None,
) -> str:
    """Generate the eval script content."""
    max_samples_arg = f"--max-samples {max_samples}" if max_samples else ""
    datasets_arg = f"--datasets {' '.join(datasets)}" if datasets else ""
    streaming_arg = "--streaming" if streaming else ""
    workers_arg = f"--num-workers {num_workers}" if num_workers > 1 else ""
    assemblyai_model_arg = f"--assemblyai-model {assemblyai_model}"
    extra_args_str = " ".join(extra_args) if extra_args else ""

    assemblyai_export = ""
    if assemblyai_api_key:
        assemblyai_export = f'export ASSEMBLYAI_API_KEY="{assemblyai_api_key}"'

    extra_exports = (
        f"{assemblyai_export}\n"
        "\n# GPU optimizations\n"
        "export CUDA_VISIBLE_DEVICES=0\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
    )
    body = f"""
cd /workspace

python -m scripts.eval.cli \\
    --model {model} \\
    {datasets_arg} \\
    {max_samples_arg} \\
    {assemblyai_model_arg} \\
    {workers_arg} \\
    {streaming_arg} \\
    --output-dir /workspace/outputs \\
    {extra_args_str}
"""
    return (
        script_preamble(hf_token, pip_packages="modelscope", extras=extra_exports)
        + body
        + script_epilogue("Evaluation", "Eval script")
    )


# Prints the pod's id. RunPod sets RUNPOD_POD_ID only in the container's main
# process (PID 1), not in SSH or tmux sessions started later.
POD_ID_COMMAND = "tr '\\0' '\\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p'"


def build_serve_script(
    hf_token: str, model: str, port: int, max_batch_size: int | None, api_key: str
) -> str:
    """Generate the `ta serve` script: the batched HTTP server on 0.0.0.0:<port>."""
    batch_arg = f"--max-batch-size {max_batch_size}" if max_batch_size else ""
    key_export = f'export TINY_AUDIO_API_KEY="{api_key}"\n' if api_key else ""
    extra_exports = (
        f"{key_export}"
        "export CUDA_VISIBLE_DEVICES=0\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
    )
    body = f"""
cd /workspace
echo "Public URL: https://$({POD_ID_COMMAND})-{port}.proxy.runpod.net"

python -m scripts.serve \\
    --model {model} \\
    --host 0.0.0.0 \\
    --port {port} \\
    {batch_arg}
"""
    return (
        script_preamble(hf_token, extras=extra_exports) + body + script_epilogue("Server", "Server")
    )


# `ta runpod deploy`'s dependency install, run once per pod by
# `runpod.install_dependencies` (which uploads it and captures its log).
INSTALL_DEPS_SCRIPT = (
    """\
#!/bin/bash
# `pipefail` ensures `pip ... | grep ...` fails when pip fails — without it,
# pip errors are masked by grep's exit code.
set -eo pipefail

export PATH="/root/.local/bin:$PATH"

"""
    + NVIDIA_LD_PATH_FIX
    + """export PIP_ROOT_USER_ACTION=ignore
export POETRY_VIRTUALENVS_CREATE=false
export PIP_BREAK_SYSTEM_PACKAGES=1

# Configure pip
mkdir -p /root/.config/pip
echo -e "[global]\\nbreak-system-packages = true" > /root/.config/pip/pip.conf

# Verify the base image actually ships a CUDA-enabled PyTorch — fail loudly
# rather than silently reinstalling a different version on top.
python -c "import torch; assert torch.cuda.is_available()" || {
    echo "ERROR: base image is missing a CUDA-enabled PyTorch. Pick a runpod/pytorch:* image." >&2
    exit 1
}

# Poetry tooling — only install what's missing
command -v poetry >/dev/null 2>&1 || pip install --user poetry
python -c "import poetry_plugin_export" 2>/dev/null || pip install --user poetry-plugin-export
poetry config virtualenvs.create false
poetry config installer.max-workers 10

# Project deps — pip skips packages already satisfied by the base image.
# `poetry export` fails fast when poetry.lock is out of sync with pyproject.toml;
# its stderr is the actionable error in that case ("Run `poetry lock` to fix").
cd /workspace
poetry export --only main --without-hashes | grep -v "^torch==" > /tmp/requirements.txt
pip install --user -r /tmp/requirements.txt

# Install project in editable mode
pip install --user -e . --no-deps

# flash-attn imports torch during its setup.py, so PEP 517 build isolation
# (the default) would pull a *different* torch into the build venv and
# compile against ABI it doesn't actually have. --no-build-isolation makes
# it build against the runpod image's torch + CUDA, which is what training
# loads. Required by configs/training/production.yaml's
# attn_implementation=flash_attention_2 — without flash-attn the model load
# falls back to sdpa with a warning.
pip install --user flash-attn --no-build-isolation --quiet

# causal-conv1d is the CUDA kernel for the depthwise causal conv inside
# Qwen3.5's linear-attention layers (three of every four layers). Without it
# transformers logs `causal_conv1d_fn` / `causal_conv1d_update` falling back to
# a reference implementation it calls "correct but much slower".
#
# This package IS the fast path: Hub kernels are not requested (see
# _load_language_model in tiny_audio/asr_modeling.py), so transformers'
# resolution order Hub -> package -> torch starts at the package. A failed
# build here means the reference path for the whole run.
#
# Installed here rather than as a project dependency for two reasons. It only
# publishes an sdist, so it compiles against nvcc and torch at install time and
# needs --no-build-isolation for the same reason flash-attn above does. And the
# `poetry export --only main` line further up would skip an optional group
# anyway — the pyproject `hybrid-kernels` group exists for local pods, but the
# bootstrap path is this script.
#
# Non-fatal: the fallback is numerically correct, so an image without nvcc
# should train slower rather than fail to deploy.
#
# ninja/packaging are declared build deps of the sdist, and --no-build-isolation
# means pip will not fetch them itself — without ninja the compile silently
# drops to a single-threaded path that takes far longer.
pip install --user ninja packaging --quiet
pip install --user causal-conv1d --no-build-isolation --quiet \
  || echo "WARN: causal-conv1d build failed; Qwen3.5 conv falls back to the slower reference path"

# flash-linear-attention is the fast path for the gated delta rule in those
# same layers (Hub kernels are not requested, so this package is what
# transformers picks first). It is only safe when paired with tilelang.
# On Hopper with
# Triton >=3.4.0 and <3.7.1 fla's Triton kernel for gated chunk_bwd_dqkwg is
# known-wrong (fla-org#640), so fla raises instead of producing bad gradients.
# Its TileLang backend is auto-enabled on exactly that combination but needs
# both the tilelang package and a usable nvcc; without them dispatch falls
# through to the Triton path and training dies at the first backward.
#
# So: install tilelang first (prebuilt manylinux wheel, no compile), then fla,
# then ask fla's own predicates whether the gated path would raise. If it
# would, remove fla so transformers uses its reference kernels -- slower, but
# correct and it actually runs. Verifying here means a bad combination fails at
# deploy time instead of twenty minutes into training.
pip install --user tilelang --quiet || echo "WARN: tilelang install failed"
pip install --user flash-linear-attention --quiet \
|| echo "WARN: flash-linear-attention install failed"
python - <<'FLA_CHECK' || pip uninstall -y flash-linear-attention fla-core >/dev/null 2>&1
import sys
try:
    from fla.utils import IS_NVIDIA_HOPPER, TRITON_ABOVE_3_4_0, TRITON_ABOVE_3_7_1
    from fla.ops.common.backends.tilelang import TileLangBackend
except Exception as e:
    print(f"fla not importable ({type(e).__name__}); nothing to verify")
    sys.exit(0)
broken_triton = IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1
if not broken_triton:
    print("fla: Triton gated path OK on this GPU/Triton combination")
    sys.exit(0)
if TileLangBackend.is_available() and TileLangBackend.is_enabled():
    print("fla: Hopper + broken Triton, but TileLang backend is active")
    sys.exit(0)
print(
    "fla: Hopper with Triton in the broken range and no usable TileLang backend "
    "-- removing flash-linear-attention so training uses the reference kernels"
)
sys.exit(1)
FLA_CHECK

# liger-kernel provides the fused linear cross-entropy used by
# apply_liger_kernel_to_qwen3() in scripts/train.py. poetry export already
# pulls it on linux, but reinstall defensively in case the editable
# project install ordering above left it behind.
pip install --user --upgrade liger-kernel --quiet

# Pre-fetch the NLTK punkt tokenizer used by truecase in scripts/labels.py's
# label normalizer. NLTK 3.9+ uses `punkt_tab` (new data package format);
# older NLTKs use `punkt`. Download both so the code works regardless of
# which NLTK version the base image ships. Doing the download here (during
# install) avoids multi-worker race on the cache path at first training step.
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

# Verify torch is available (from base image) — we never pin or replace torch.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Verify the user-site install actually landed where `ta` will look.
# `/root/.local/bin/ta` is a generated console script with a shebang pinned
# to the python pip used; if its interpreter can't import typer, then
# pip --user installed to a different python's user-site than the one ta
# runs under (e.g., python3.10 vs python3.11 in the base image), and
# every subsequent `ta dev <cmd>` will fail with `ModuleNotFoundError`.
TA_PYTHON=$(head -1 /root/.local/bin/ta | sed 's|^#!||')
if ! "$TA_PYTHON" -c "import typer, hydra, omegaconf, datasets, transformers, truecase, ftfy" \
2>/tmp/tiny_audio_import_check.err; then
    echo "ERROR: deps did not install into the python that /root/.local/bin/ta uses." >&2
    echo "  ta interpreter: $TA_PYTHON" >&2
    echo "  pip used:        $(which pip) ($(pip --version))" >&2
    echo "  python --user site: $(python -m site --user-site)" >&2
    echo "  ta-python --user site: $("$TA_PYTHON" -m site --user-site 2>&1)" >&2
    cat /tmp/tiny_audio_import_check.err >&2
    exit 1
fi
echo "Dependencies verified for $TA_PYTHON"
"""
)
