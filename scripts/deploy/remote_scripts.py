"""Shell scripts run inside tmux on a RunPod pod for training and evaluation.

Pure string builders: they take the run parameters and return the bash text
`ta runpod train` / `ta runpod eval` upload and execute.

Every caller-supplied value is spliced in through shlex.quote / shlex.join.
The script reaches the pod over SFTP and tmux runs it as a file, so bash
parsing the script is the only shell layer: one level of quoting is exactly
right, and it keeps a token or key containing `$`, a backtick or a quote, or
a Hydra override containing `[`, `*` or spaces, from being expanded or split.
"""

from __future__ import annotations

import shlex

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
export HF_TOKEN={hf_token}
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
        hf_token: Value exported as HF_TOKEN (shell-quoted here).
        pip_packages: Extra packages to install before the run; the pip line is
            omitted entirely when empty. Only the eval script needs one
            (modelscope) now that Xet has replaced hf_transfer.
        extras: Extra `export` lines appended to the header.
    """
    pip_install = (
        f"pip install {pip_packages} --quiet --root-user-action=ignore\n" if pip_packages else ""
    )
    preamble = _SCRIPT_PREAMBLE.format(pip_install=pip_install, hf_token=shlex.quote(hf_token))
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
        wandb_exports += f"export WANDB_RUN_ID={shlex.quote(wandb_run_id)}\n"
    if wandb_resume:
        wandb_exports += f"export WANDB_RESUME={shlex.quote(wandb_resume)}\n"

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
    extra_args_str = shlex.join(extra_args)
    extra_exports = training_exports(wandb_run_id, wandb_resume)
    body = f"""
cd /workspace
python -m scripts.train {shlex.quote(f"+experiments={experiment}")} {extra_args_str}"""
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
    datasets_arg = f"--datasets {shlex.join(datasets)}" if datasets else ""
    streaming_arg = "--streaming" if streaming else ""
    workers_arg = f"--num-workers {num_workers}" if num_workers > 1 else ""
    assemblyai_model_arg = f"--assemblyai-model {shlex.quote(assemblyai_model)}"
    extra_args_str = shlex.join(extra_args or [])

    assemblyai_export = ""
    if assemblyai_api_key:
        assemblyai_export = f"export ASSEMBLYAI_API_KEY={shlex.quote(assemblyai_api_key)}"

    extra_exports = (
        f"{assemblyai_export}\n"
        "\n# GPU optimizations\n"
        "export CUDA_VISIBLE_DEVICES=0\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
    )
    body = f"""
cd /workspace

python -m scripts.eval.cli \\
    --model {shlex.quote(model)} \\
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
