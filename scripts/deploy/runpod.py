#!/usr/bin/env python3
"""Unified CLI for RunPod operations."""

import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from fabric import Connection
from invoke import UnexpectedExit

from scripts.utils import get_project_root

app = typer.Typer(help="RunPod remote operations CLI")

SSH_KEY_PATH = "~/.ssh/id_ed25519"


def _auto_session_name(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M")
    return f"{prefix}_{timestamp}"


def _start_remote_tmux_script(
    conn: Connection,
    host: str,
    port: int,
    session_name: str,
    script_content: str,
    script_path: str,
    no_attach: bool,
) -> None:
    """Upload a script to /tmp, start it in a tmux session, optionally attach."""
    conn.run(f"cat > {script_path} << 'EOF'\n{script_content}\nEOF", hide=True)
    conn.run(f"chmod +x {script_path}", hide=True)
    result = conn.run(f"tmux new-session -d -s {session_name} {script_path}", warn=True)
    if not result.ok:
        print(f"Failed to start tmux session: {result.stderr}")
        sys.exit(1)
    print(f"\nSession '{session_name}' started.")
    print(f"To re-attach later: ssh -p {port} root@{host} -t 'tmux attach -t {session_name}'")
    if not no_attach:
        time.sleep(2)
        attach_tmux_session(host, port, session_name)


def get_connection(host: str, port: int) -> Connection:
    """Create a Fabric connection with standard SSH settings."""
    return Connection(
        host=host,
        user="root",
        port=port,
        connect_kwargs={
            "key_filename": str(Path(SSH_KEY_PATH).expanduser()),
            "look_for_keys": False,
            "allow_agent": False,
        },
        connect_timeout=10,
    )


def test_connection(conn: Connection) -> bool:
    """Test SSH connection to the remote host."""
    print(f"Testing SSH connection to {conn.host}:{conn.port}...")
    try:
        conn.run("echo Connected", hide=True)
    except Exception as e:
        print(f"Failed to connect via SSH: {e}")
        return False
    print("SSH connection successful!")
    return True


def list_tmux_sessions(conn: Connection) -> list[str]:
    """Get list of tmux session names on remote host."""
    try:
        result = conn.run('tmux list-sessions -F "#S" 2>/dev/null', hide=True, warn=True)
        if result.ok and result.stdout.strip():
            return result.stdout.strip().split("\n")
    except UnexpectedExit:
        pass
    return []


def kill_tmux_session(conn: Connection, session_name: str) -> bool:
    """Kill a tmux session by name. Returns True if killed, False if not found."""
    result = conn.run(f"tmux kill-session -t {session_name}", hide=True, warn=True)
    return result.ok


def get_tmux_logs(conn: Connection, session_name: str, lines: int = 100) -> str | None:
    """Capture recent output from a tmux session."""
    try:
        result = conn.run(
            f"tmux capture-pane -t '{session_name}' -p -S -{lines}",
            hide=True,
            warn=True,
        )
        if result.ok:
            return result.stdout
    except UnexpectedExit:
        pass
    return None


def attach_tmux_session(host: str, port: int, session_name: str) -> None:
    """Attach to a tmux session interactively (requires subprocess for TTY)."""
    print(f"\nAttaching to session '{session_name}'...")
    print("=" * 50)
    print("TMUX CONTROLS:")
    print("  - Detach (and leave running): Ctrl+B then D")
    print("  - Scroll Mode:              Ctrl+B then [ (use arrows, q to exit)")
    print("=" * 50)

    cmd = (
        f"ssh -i {SSH_KEY_PATH} -p {port} -o StrictHostKeyChecking=no "
        f"-t root@{host} \"tmux attach-session -t '{session_name}'\""
    )
    subprocess.run(cmd, shell=True, check=False)
    print(f"\nDetached from session '{session_name}'.")


# Suffixes filtered out of the rsync file set on top of gitignore. Belt-and-
# suspenders against accidentally syncing checkpoint weights into a RunPod
# workspace.
RSYNC_SUFFIX_BLOCKLIST = (".safetensors",)


def _gitignore_aware_file_list(project_root: Path) -> str:
    """Return newline-separated repo-relative paths git would track or add.

    Uses ``git ls-files --cached --others --exclude-standard`` so the rsync
    file set has exact gitignore semantics (including ``!`` un-ignore lines,
    which rsync's own ``:- .gitignore`` filter mishandles). Suffixes in
    ``RSYNC_SUFFIX_BLOCKLIST`` are then dropped on top.
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    # Drop tracked-but-deleted paths — they're in --cached but missing on disk,
    # so rsync --files-from would skip them and exit 23.
    deleted = subprocess.run(
        ["git", "ls-files", "--deleted"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    deleted_set = set(deleted.stdout.splitlines())
    lines = [
        line
        for line in result.stdout.splitlines()
        if line and line not in deleted_set and not line.endswith(RSYNC_SUFFIX_BLOCKLIST)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def setup_remote_environment(conn: Connection) -> None:
    """Install system dependencies the base RunPod image is missing.

    apt-get install is idempotent, so this is a no-op when packages are already
    present in the image. Anything the image already ships (PyTorch, CUDA,
    Python, common libs) we leave alone.
    """
    print("\nSetting up remote environment...")
    conn.run("apt-get update -qq || true")
    conn.run(
        "apt-get install -y -qq ffmpeg tmux rsync libsndfile1",
    )
    # portaudio19-dev is only needed for pyaudio runtime, never training;
    # fall back gracefully on pods where it isn't available.
    conn.run(
        "apt-get install -y portaudio19-dev || true",
    )
    print("Remote environment setup complete!")


def sync_project(conn: Connection, project_root: Path) -> None:
    """Sync project files to the RunPod instance using rsync.

    File set comes from git so gitignore is honored exactly. ``--delete`` is
    omitted (it doesn't compose cleanly with ``--files-from`` and would
    require a parallel exclude set anyway); stale files on the remote are
    harmless for training. Wipe ``/workspace`` manually if needed.
    """
    print(f"\nSyncing project from {project_root}...")

    file_list = _gitignore_aware_file_list(project_root)
    if not file_list.strip():
        raise RuntimeError(f"git ls-files returned no files under {project_root}")

    rsync_cmd = (
        f"rsync -avz --no-owner --no-group --files-from=- "
        f'-e "ssh -i ~/.ssh/id_ed25519 -p {conn.port} -o StrictHostKeyChecking=no" '
        f"{project_root}/ root@{conn.host}:/workspace/"
    )

    subprocess.run(rsync_cmd, shell=True, check=True, input=file_list, text=True)
    print("Project synced successfully!")


def install_dependencies(conn: Connection) -> None:
    """Install Python dependencies on top of the base RunPod image.

    Trusts the base image to provide a CUDA-enabled PyTorch and Python; we only
    fill in the gaps (Poetry tooling + project deps).
    """
    print("\nInstalling Python dependencies...")

    setup_script = """\
#!/bin/bash
# `pipefail` ensures `pip ... | grep ...` fails when pip fails — without it,
# pip errors are masked by grep's exit code.
set -eo pipefail

export PATH="/root/.local/bin:$PATH"

# RunPod images ship torch in system dist-packages alongside its nvidia-*
# CUDA wheels. When the project pins a different torch version it installs
# into --user and shadows the image copy, but the nvidia libs stay in the
# system tree -- so the loader cannot find e.g. libcusparseLt.so.0 and every
# `import torch` dies with ImportError. Observed on
# runpod/pytorch:...-torch291 against this repo's torch ~2.8.0 pin; it also
# broke the flash-attn build, whose metadata hook imports torch.
NVLIBS="$(python3 -c 'import glob;print(":".join(sorted(glob.glob("/usr/local/lib/python*/dist-packages/nvidia/*/lib"))))')"
# Spelled out with if/else on purpose: these scripts are built with Python
# f-strings, so shell brace-expansion syntax would be parsed as an f-string
# replacement field and raise NameError at build time.
if [ -n "$LD_LIBRARY_PATH" ]; then
  export LD_LIBRARY_PATH="$NVLIBS:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$NVLIBS"
fi
export PIP_ROOT_USER_ACTION=ignore
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
# same layers, but it is only safe when paired with tilelang. On Hopper with
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
pip install --user flash-linear-attention --quiet || echo "WARN: flash-linear-attention install failed"
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

# Pre-fetch the NLTK punkt tokenizer used by truecase in scripts/train.py's
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
if ! "$TA_PYTHON" -c "import typer, hydra, omegaconf, datasets, transformers, truecase, ftfy" 2>/tmp/tiny_audio_import_check.err; then
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

    # Upload the script via a single-quoted heredoc so apostrophes, dollar
    # signs, and other shell metachars in the body are preserved verbatim.
    # This avoids the entire class of `bash -c '...'` quoting bugs.
    script_path = "/tmp/tiny_audio_install_deps.sh"
    log_path = "/tmp/tiny_audio_install.log"
    conn.run(
        f"cat > {script_path} << 'INSTALL_DEPS_EOF'\n{setup_script}\nINSTALL_DEPS_EOF",
        hide=True,
    )
    # Capture all output to a log file silently rather than streaming live.
    # Pip's progress bars + ANSI color codes corrupt the local TTY when piped
    # through Fabric. On failure we fetch the tail and print it as plain text.
    print(f"  (silent; remote log: {log_path})")
    try:
        conn.run(f"bash {script_path} > {log_path} 2>&1", hide=True)
    except UnexpectedExit:
        print(f"\n[install_dependencies] FAILED. Last 80 lines of {log_path}:\n")
        tail = conn.run(f"tail -n 80 {log_path}", hide=True, warn=True)
        sys.stdout.write(tail.stdout)
        sys.stdout.flush()
        raise
    print(f"Dependencies installed successfully! Full log: {log_path}")


@app.command(name="plan")
def plan(
    experiment: str = typer.Option("granite_gemma", "--experiment", "-e"),
    seq_len: int = typer.Option(512, "--seq-len", help="Assumed tokens per sample"),
    gpu: str = typer.Option("NVIDIA H100 80GB HBM3", "--gpu"),
    image: str = typer.Option("runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404", "--image"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    overrides: list[str] = typer.Argument(None, help="Extra Hydra overrides"),
):
    """Estimate GPU memory + disk for a config and emit a pod create command."""
    from scripts.deploy.plan import plan_command

    plan_command(
        experiment=experiment,
        seq_len=seq_len,
        gpu=gpu,
        image=image,
        as_json=as_json,
        overrides=overrides,
    )


@app.command(name="up")
def up(
    experiment: str = typer.Option("granite_gemma", "--experiment", "-e"),
    seq_len: int = typer.Option(512, "--seq-len"),
    name: str | None = typer.Option(None, "--name"),
    image: str = typer.Option("runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404", "--image"),
    max_attempts: int = typer.Option(6, "--max-attempts"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    overrides: list[str] = typer.Argument(None),
):
    """Size a config, then create a pod on the first GPU type with capacity."""
    from scripts.deploy.plan import provision_command

    provision_command(
        experiment=experiment,
        seq_len=seq_len,
        name=name,
        image=image,
        max_attempts=max_attempts,
        dry_run=dry_run,
        overrides=overrides,
    )


@app.command(name="wait")
def wait(
    pod_id: str = typer.Argument(..., help="Pod id from `ta runpod up`"),
    timeout_s: int = typer.Option(900, "--timeout"),
):
    """Block until a pod exposes SSH, then print `<ip> <port>`."""
    from scripts.deploy.plan import wait_command

    wait_command(pod_id=pod_id, timeout_s=timeout_s)


@app.command()
def deploy(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    skip_setup: bool = typer.Option(False, "--skip-setup", help="Skip remote environment setup"),
    skip_sync: bool = typer.Option(False, "--skip-sync", help="Skip project file sync"),
    skip_deps: bool = typer.Option(
        False, "--skip-deps", help="Skip Python dependency installation"
    ),
):
    """Deploy ASR project to a RunPod instance."""
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    project_root = get_project_root()

    if not skip_setup:
        setup_remote_environment(conn)

    if not skip_sync:
        sync_project(conn, project_root)

    if not skip_deps:
        install_dependencies(conn)

    print("\nDeployment finished!")
    print(f"To connect: ssh -i ~/.ssh/id_ed25519 -p {port} root@{host}")


# Every remote script shares this header: the fd limit, the nvidia-lib
# LD_LIBRARY_PATH repair, and the HF cache/token exports. It lived inline in all
# three builders below and had already drifted between them.
_SCRIPT_PREAMBLE = """#!/bin/bash
# NOTE: "set -e" intentionally removed so session stays active on crash for debugging

ulimit -n 65536
{pip_install}export PATH="/root/.local/bin:$PATH"

# RunPod images ship torch in system dist-packages alongside its nvidia-*
# CUDA wheels. When the project pins a different torch version it installs
# into --user and shadows the image copy, but the nvidia libs stay in the
# system tree -- so the loader cannot find e.g. libcusparseLt.so.0 and every
# `import torch` dies with ImportError. Observed on
# runpod/pytorch:...-torch291 against this repo's torch ~2.8.0 pin; it also
# broke the flash-attn build, whose metadata hook imports torch.
NVLIBS="$(python3 -c 'import glob;print(":".join(sorted(glob.glob("/usr/local/lib/python*/dist-packages/nvidia/*/lib"))))')"
# Spelled out with if/else on purpose: these scripts are built with Python
# f-strings, so shell brace-expansion syntax would be parsed as an f-string
# replacement field and raise NameError at build time.
if [ -n "$LD_LIBRARY_PATH" ]; then
  export LD_LIBRARY_PATH="$NVLIBS:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$NVLIBS"
fi
export HF_HOME=/workspace/.cache/huggingface
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


def _script_preamble(hf_token: str, *, pip_packages: str = "", extras: str = "") -> str:
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


def _script_epilogue(label: str, finished: str) -> str:
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


def build_training_script(
    experiment: str,
    hf_token: str,
    wandb_run_id: str | None,
    wandb_resume: str | None,
    extra_args: list[str],
) -> str:
    """Generate the training script content."""
    wandb_exports = ""
    if wandb_run_id:
        wandb_exports += f'export WANDB_RUN_ID="{wandb_run_id}"\n'
    if wandb_resume:
        wandb_exports += f'export WANDB_RESUME="{wandb_resume}"\n'

    extra_args_str = " ".join(extra_args) if extra_args else ""

    extra_exports = (
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
    body = f"""
cd /workspace
python -m scripts.train +experiments={experiment} {extra_args_str}"""
    return (
        _script_preamble(hf_token, extras=extra_exports)
        + body
        + _script_epilogue("Training", "Training script")
    )


def _remote_free_gib(conn: Connection, path: str = "/workspace") -> float | None:
    """Free space on the filesystem backing `path`, in GiB (None if unreadable)."""
    result = conn.run(f"df -Pk {path} | tail -1", hide=True, warn=True)
    if not result.ok:
        return None
    fields = result.stdout.split()
    try:
        # `df -Pk` reports 1K blocks; the 4th field is Available.
        return int(fields[3]) / (1024**2)
    except (IndexError, ValueError):
        return None


# Where a run materializes its bulk: the dataset cache (parquet + generated
# arrow) and the Hub cache (model weights). Both are exported by
# _script_preamble, so they are the same paths the training script will use.
_REMOTE_CACHE_DIRS = ("/workspace/datasets", "/workspace/.cache/huggingface")


def _remote_used_gib(conn: Connection, paths: tuple[str, ...]) -> float:
    """GiB already materialized under `paths` on the pod (0 if unreadable).

    `du` walks metadata, and /workspace is often a network volume holding a
    few hundred thousand parquet shards, so each call is bounded by `timeout`
    rather than allowed to stall the preflight. An unreadable or missing path
    contributes 0, which just makes the check conservative again.
    """
    total_kb = 0
    for path in paths:
        result = conn.run(f"timeout 60 du -sk {path} 2>/dev/null | tail -1", hide=True, warn=True)
        if not result.ok or not result.stdout.strip():
            continue
        try:
            total_kb += int(result.stdout.split()[0])
        except (IndexError, ValueError):
            continue
    return total_kb / (1024**2)


def _check_remote_disk(conn: Connection, experiment: str, overrides: list[str]) -> None:
    """Refuse to start a run that /workspace cannot physically hold.

    ENOSPC does not surface at launch. It surfaces hours later as an xet
    "File reconstruction error: No space left on device" partway through
    dataset prep, after the pod has been billed for the whole time. The plan
    already knows what a recipe needs and `df` knows what the pod has, so
    there is no reason to find out the expensive way. The usual trigger is a
    pod sized for one experiment then handed a different one to train --
    `up` and `train` share a default, but either can be pointed elsewhere
    with -e.
    """
    from scripts.deploy.plan import build_plan

    free = _remote_free_gib(conn)
    if free is None:
        print("Could not read `df /workspace`; skipping the disk preflight.")
        return
    try:
        need = build_plan(experiment, overrides, 512).disk["recommended"]
    except Exception as exc:  # unresolvable config, gated repo, Hub outage
        print(f"Disk preflight skipped ({type(exc).__name__}: {exc}).")
        return

    # `need` is the size of a run starting from an empty pod. Re-running an
    # experiment on a pod that already holds its dataset would double-count:
    # the bytes are simultaneously "required" and already subtracted from
    # `free`. Compare against what is still left to write instead.
    cached = _remote_used_gib(conn, _REMOTE_CACHE_DIRS)
    remaining = max(need - cached, 0.0)
    print(
        f"/workspace has {free:,.0f} GiB free; {experiment} needs ~{need:,.0f} GiB total, "
        f"~{cached:,.0f} GiB already cached -> ~{remaining:,.0f} GiB still to write."
    )
    if free < remaining:
        print(
            f"\nNot enough disk -- short by {remaining - free:,.0f} GiB. `datasets` holds\n"
            "the downloaded parquet and its generated arrow tables at the same time, and\n"
            "checkpoints scale with the trainable stack, so this run would die with\n"
            "ENOSPC mid-prep. Levers, cheapest first: lower `save_total_limit`, provision\n"
            f"a bigger pod (see `ta runpod plan -e {experiment}`), or pass\n"
            "--skip-disk-check to override."
        )
        raise typer.Exit(1)


@app.command()
def train(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    experiment: str = typer.Option(
        "granite_gemma", "--experiment", "-e", help="Experiment config to run"
    ),
    session_name: str | None = typer.Option(
        None, "--session-name", "-s", help="Custom tmux session name"
    ),
    no_attach: bool = typer.Option(False, "--no-attach", help="Start session but don't attach"),
    force: bool = typer.Option(False, "--force", "-f", help="Kill existing session with same name"),
    skip_disk_check: bool = typer.Option(
        False, "--skip-disk-check", help="Start even if /workspace looks too small"
    ),
    wandb_run_id: str | None = typer.Option(None, "--wandb-run-id", help="W&B run ID to resume"),
    wandb_resume: Annotated[
        str | None,
        typer.Option("--wandb-resume", help="W&B resume mode: must, allow, or never"),
    ] = None,
    extra_args: Annotated[
        list[str] | None,
        typer.Argument(help="Extra Hydra overrides passed to training script"),
    ] = None,
):
    """Start training on a remote RunPod instance in a tmux session."""
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    if not skip_disk_check:
        _check_remote_disk(conn, experiment, list(extra_args or []))

    if session_name is None:
        session_name = _auto_session_name(f"train_{experiment}")

    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)

    hf_token = os.environ.get("HF_TOKEN", "")
    wandb_run_id = wandb_run_id or os.environ.get("WANDB_RUN_ID")
    wandb_resume = wandb_resume or os.environ.get("WANDB_RESUME")

    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")

    print(f"\nStarting training session '{session_name}' with experiment '{experiment}'...")
    if extra_args:
        print(f"Extra args: {' '.join(extra_args)}")

    _start_remote_tmux_script(
        conn,
        host,
        port,
        session_name,
        build_training_script(experiment, hf_token, wandb_run_id, wandb_resume, extra_args or []),
        f"/tmp/train_{session_name}.sh",
        no_attach,
    )


@app.command()
def attach(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    session_name: str | None = typer.Option(None, "--session-name", "-s", help="Tmux session name"),
    list_sessions: bool = typer.Option(False, "--list", "-l", help="List all sessions and exit"),
    logs: bool = typer.Option(False, "--logs", help="Show recent logs instead of attaching"),
    lines: int = typer.Option(100, "--lines", "-n", help="Number of log lines to show"),
):
    """Attach to, list, or view logs from a tmux session on a remote RunPod instance."""
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    sessions = list_tmux_sessions(conn)

    if list_sessions:
        print("\nAvailable tmux sessions:")
        if not sessions:
            print("  No active sessions found.")
        else:
            for session in sessions:
                print(f"  - {session}")
        return

    if not session_name:
        if not sessions:
            print("\nNo active tmux sessions found. Start a training session first.")
            sys.exit(1)
        elif len(sessions) == 1:
            session_name = sessions[0]
            print(f"\nFound one active session: '{session_name}'. Proceeding automatically.")
        else:
            print("\nMultiple active sessions found. Please choose one:")
            for i, session in enumerate(sessions, 1):
                print(f"  {i}. {session}")
            try:
                choice = input(f"Enter number (1-{len(sessions)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    session_name = sessions[idx]
                else:
                    print("Invalid selection. Exiting.")
                    sys.exit(1)
            except (KeyboardInterrupt, ValueError):
                print("\nSelection cancelled. Exiting.")
                sys.exit(0)

    if logs:
        log_content = get_tmux_logs(conn, session_name, lines)
        if log_content:
            print("\n" + "=" * 50)
            print(log_content)
            print("=" * 50)
        else:
            print(f"Session '{session_name}' not found or an error occurred.")
    else:
        attach_tmux_session(host, port, session_name)


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
        _script_preamble(hf_token, pip_packages="modelscope", extras=extra_exports)
        + body
        + _script_epilogue("Evaluation", "Eval script")
    )


@app.command("eval")
def eval_model(
    host: str = typer.Argument(..., help="RunPod instance IP address or hostname"),
    port: int = typer.Argument(..., help="SSH port for the RunPod instance"),
    model: str = typer.Option(..., "--model", "-m", help="Model path/ID or 'assemblyai'"),
    datasets: Annotated[
        list[str] | None,
        typer.Option("--datasets", "-d", help="Datasets to evaluate on"),
    ] = None,
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-n", help="Max samples per dataset"
    ),
    assemblyai_model: str = typer.Option(
        "universal-3-pro",
        "--assemblyai-model",
        help="AssemblyAI model (best, universal, universal-3-pro)",
    ),
    num_workers: int = typer.Option(
        1, "--num-workers", "-w", help="Number of parallel workers for API evaluations"
    ),
    streaming: bool = typer.Option(False, "--streaming", "-s", help="Use streaming evaluation"),
    session_name: str | None = typer.Option(
        None, "--session-name", help="Custom tmux session name"
    ),
    no_attach: bool = typer.Option(False, "--no-attach", help="Start session but don't attach"),
    force: bool = typer.Option(False, "--force", "-f", help="Kill existing session with same name"),
    extra_args: Annotated[
        list[str] | None,
        typer.Argument(help="Extra arguments passed to eval script"),
    ] = None,
):
    """Run ASR evaluation on a remote RunPod instance.

    Examples:
        runpod eval host port -m mazesmazes/tiny-audio -d loquacious
        runpod eval host port -m assemblyai --assemblyai-model universal-3-pro -d loquacious -w 4
    """
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    if session_name is None:
        model_short = model.rsplit("/", maxsplit=1)[-1] if "/" in model else model
        session_name = _auto_session_name(f"eval_{model_short}")

    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)

    hf_token = os.environ.get("HF_TOKEN", "")
    assemblyai_api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")

    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")
    if model == "assemblyai" and not assemblyai_api_key:
        print(
            "Warning: ASSEMBLYAI_API_KEY environment variable not set (required for assemblyai model)."
        )

    if datasets is None:
        datasets = ["loquacious"]

    print(f"\nStarting eval session '{session_name}'...")
    print(f"Model: {model}")
    print(f"Datasets: {', '.join(datasets)}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    if num_workers > 1:
        print(f"Workers: {num_workers}")
    if streaming:
        print("Streaming mode enabled")

    _start_remote_tmux_script(
        conn,
        host,
        port,
        session_name,
        build_eval_script(
            hf_token,
            model,
            datasets,
            max_samples,
            assemblyai_api_key,
            assemblyai_model,
            num_workers,
            streaming,
            extra_args,
        ),
        f"/tmp/eval_{session_name}.sh",
        no_attach,
    )


@app.command()
def checkpoint(
    host: str = typer.Argument(..., help="Remote server IP or hostname"),
    port: int = typer.Argument(22, help="SSH port"),
):
    """Find the latest checkpoint on a remote training server."""
    conn = get_connection(host, port)

    if not test_connection(conn):
        sys.exit(1)

    result = conn.run(
        "find /workspace/outputs -name 'checkpoint-*' -type d 2>/dev/null | sort -V | tail -1",
        hide=True,
        warn=True,
    )

    if not result.ok:
        print(f"Error: {result.stderr}", file=sys.stderr)
        raise typer.Exit(1)

    ckpt = result.stdout.strip()
    if ckpt:
        print(ckpt)
    else:
        print("No checkpoints found", file=sys.stderr)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
