#!/usr/bin/env python3
"""Unified CLI for RunPod operations."""

import io
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from fabric import Connection
from invoke.exceptions import UnexpectedExit
from rich.prompt import Prompt
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed

from scripts.deploy import plan as deploy_plan
from scripts.deploy import pods, remote_scripts
from scripts.deploy.pods import SERVE_PORT
from scripts.eval.constants import DEFAULT_DATASET, AssemblyAIModel, get_model_name
from scripts.utils import get_project_root

app = typer.Typer(help="Train and evaluate on remote RunPod pods.", add_completion=False)

SSH_KEY_PATH = "~/.ssh/id_ed25519"
SSH_CONNECT_ATTEMPTS = 3
SSH_CONNECT_WAIT_S = 5

# Every command that talks to a pod takes the same two positionals, spelled and
# described identically, so `ta runpod <cmd> <HOST> <PORT>` always works.
HostArg = Annotated[str, typer.Argument(help="RunPod instance IP address or hostname")]
PortArg = Annotated[int, typer.Argument(help="SSH port for the RunPod instance")]

# Shared option help, so sibling commands describe the same flag the same way.
EXPERIMENT_HELP = "Experiment config under configs/experiments/ to run"
SEQ_LEN_HELP = "Assumed tokens per sample when sizing memory"
IMAGE_HELP = "RunPod container image"
OVERRIDES_HELP = "Extra Hydra overrides (key=value)"
SESSION_NAME_HELP = "tmux session name (default: derived from the command)"
NO_ATTACH_HELP = "Start the session but don't attach to it"
FORCE_HELP = "Kill an existing session with the same name first"
HF_TOKEN_HELP = "Hugging Face token exported to the pod as HF_TOKEN"
DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"


def _auto_session_name(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M")
    return f"{prefix}_{timestamp}"


def _put_text(conn: Connection, text: str, remote: str) -> None:
    """Upload a string to `remote` over SFTP.

    Encode to bytes ourselves rather than handing `put` an `io.StringIO`.
    Paramiko's `putfo` sizes the transfer by summing `len(chunk)` over reads
    from the file object, then (with confirm=True) stats the remote path and
    compares. Reading from a StringIO yields *characters* while the wire
    carries *UTF-8 bytes*, so a single non-ASCII character in a script body --
    one em dash in a comment is enough -- makes the two disagree and raises
    "size mismatch in put!" on a transfer that in fact succeeded.
    """
    conn.put(io.BytesIO(text.encode("utf-8")), remote=remote)


def _prepare_session(conn: Connection, session_name: str, force: bool, hf_token: str) -> str:
    """Shared launch preamble: optionally kill a same-named session, warn on no token."""
    if force:
        print(f"Killing existing session '{session_name}' if present...")
        kill_tmux_session(conn, session_name)
    if not hf_token:
        print("Warning: HF_TOKEN is not set; the pod will not be able to pull gated repos.")
    return session_name


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
    # SFTP upload: no shell in the path, so the script body needs no quoting.
    _put_text(conn, script_content, script_path)
    conn.sftp().chmod(script_path, 0o700)
    result = conn.run(
        f"tmux new-session -d -s {shlex.quote(session_name)} {shlex.quote(script_path)}",
        warn=True,
    )
    if not result.ok:
        print(f"Failed to start tmux session: {result.stderr}")
        raise typer.Exit(1)
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

    # `ta runpod wait` returns once RunPod assigns the SSH endpoint, which can be
    # a few seconds before sshd inside the container accepts connections.
    @retry(stop=stop_after_attempt(SSH_CONNECT_ATTEMPTS), wait=wait_fixed(SSH_CONNECT_WAIT_S))
    def probe() -> None:
        conn.run("echo Connected", hide=True)

    try:
        probe()
    except RetryError as e:
        print(f"Failed to connect via SSH: {e.last_attempt.exception()}")
        return False
    print("SSH connection successful!")
    return True


def connect(host: str, port: int) -> Connection:
    """Open and verify the SSH connection every pod command starts with."""
    conn = get_connection(host, port)
    if not test_connection(conn):
        raise typer.Exit(1)
    return conn


def list_tmux_sessions(conn: Connection) -> list[str]:
    """Get list of tmux session names on remote host."""
    result = conn.run('tmux list-sessions -F "#S" 2>/dev/null', hide=True, warn=True)
    if result.ok and result.stdout.strip():
        return result.stdout.strip().split("\n")
    return []


def kill_tmux_session(conn: Connection, session_name: str) -> bool:
    """Kill a tmux session by name. Returns True if killed, False if not found."""
    result = conn.run(f"tmux kill-session -t {shlex.quote(session_name)}", hide=True, warn=True)
    return result.ok


def get_tmux_logs(conn: Connection, session_name: str, lines: int = 100) -> str | None:
    """Capture recent output from a tmux session."""
    result = conn.run(
        f"tmux capture-pane -t {shlex.quote(session_name)} -p -S -{lines}",
        hide=True,
        warn=True,
    )
    return result.stdout if result.ok else None


def attach_tmux_session(host: str, port: int, session_name: str) -> None:
    """Attach to a tmux session interactively (requires subprocess for TTY)."""
    print(f"\nAttaching to session '{session_name}'...")
    print("=" * 50)
    print("TMUX CONTROLS:")
    print("  - Detach (and leave running): Ctrl+B then D")
    print("  - Scroll Mode:              Ctrl+B then [ (use arrows, q to exit)")
    print("=" * 50)

    subprocess.run(
        [
            "ssh",
            "-i",
            str(Path(SSH_KEY_PATH).expanduser()),
            "-p",
            str(port),
            "-o",
            "StrictHostKeyChecking=no",
            "-t",
            f"root@{host}",
            f"tmux attach-session -t {shlex.quote(session_name)}",
        ],
        check=False,
    )
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
    lines = [
        line
        for line in result.stdout.splitlines()
        if line and not line.endswith(RSYNC_SUFFIX_BLOCKLIST) and (project_root / line).exists()
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
        msg = f"git ls-files returned no files under {project_root}"
        raise RuntimeError(msg)

    # argv form: nothing here needs a shell. rsync splits the `-e` command on
    # whitespace itself, so it stays one argument.
    ssh_command = (
        f"ssh -i {Path(SSH_KEY_PATH).expanduser()} -p {conn.port} -o StrictHostKeyChecking=no"
    )
    subprocess.run(
        [
            "rsync",
            "-avz",
            "--no-owner",
            "--no-group",
            "--files-from=-",
            "-e",
            ssh_command,
            f"{project_root}/",
            f"root@{conn.host}:/workspace/",
        ],
        check=True,
        input=file_list,
        text=True,
    )
    print("Project synced successfully!")


def install_dependencies(conn: Connection) -> None:
    """Install Python dependencies on top of the base RunPod image.

    Trusts the base image to provide a CUDA-enabled PyTorch and Python; we only
    fill in the gaps (Poetry tooling + project deps).
    """
    print("\nInstalling Python dependencies...")

    setup_script = remote_scripts.INSTALL_DEPS_SCRIPT

    # Upload over SFTP so apostrophes, dollar signs, and other shell metachars
    # in the body are preserved verbatim without any quoting.
    script_path = "/tmp/tiny_audio_install_deps.sh"
    log_path = "/tmp/tiny_audio_install.log"
    _put_text(conn, setup_script, script_path)
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
    experiment: Annotated[
        str, typer.Option("--experiment", "-e", help=EXPERIMENT_HELP)
    ] = "granite_qwen_frozen",
    # 320, not 512: the measured granite_qwen sequence is 237 audio tokens at
    # the 19s collator ceiling plus prompt and transcript. This default
    # shadows plan_command's own, so the two have to be kept in sync -- it was
    # 512 here while plan.py said 320, which silently inflated every estimate
    # by 1.6x. Same shape of bug as `attn_implementation` being set under
    # `model:` while `training:` quietly won.
    seq_len: Annotated[int, typer.Option("--seq-len", help=SEQ_LEN_HELP)] = 320,
    # No default: plan_command picks the cheapest listed GPU that actually
    # fits the estimate. Hardcoding an H100 here meant every plan recommended
    # an 80 GB card regardless of need -- granite_qwen_lora wants ~36 GiB.
    gpu: Annotated[
        str | None, typer.Option("--gpu", help="Override the GPU id (default: cheapest that fits)")
    ] = None,
    image: Annotated[str, typer.Option("--image", help=IMAGE_HELP)] = DEFAULT_IMAGE,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
    overrides: Annotated[list[str] | None, typer.Argument(help=OVERRIDES_HELP)] = None,
) -> None:
    """Estimate GPU memory + disk for a config and emit a pod create command."""
    deploy_plan.plan_command(
        experiment=experiment,
        seq_len=seq_len,
        gpu=gpu,
        image=image,
        as_json=as_json,
        overrides=overrides,
    )


@app.command(name="up")
def up(
    experiment: Annotated[
        str, typer.Option("--experiment", "-e", help=EXPERIMENT_HELP)
    ] = "granite_qwen_frozen",
    seq_len: Annotated[int, typer.Option("--seq-len", help=SEQ_LEN_HELP)] = 512,
    name: Annotated[
        str | None, typer.Option("--name", help="Pod name (default: derived from the experiment)")
    ] = None,
    image: Annotated[str, typer.Option("--image", help=IMAGE_HELP)] = DEFAULT_IMAGE,
    max_attempts: Annotated[
        int, typer.Option("--max-attempts", help="GPU types to try before giving up")
    ] = 6,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the pod create command without running it")
    ] = False,
    serve: Annotated[
        bool,
        typer.Option(
            "--serve",
            help="Create an inference pod for `ta runpod serve` instead of sizing a training run",
        ),
    ] = False,
    overrides: Annotated[list[str] | None, typer.Argument(help=OVERRIDES_HELP)] = None,
) -> None:
    """Size a config, then create a pod on the first GPU type with capacity."""
    if serve:
        pods.provision_serve_command(
            name=name, image=image, max_attempts=max_attempts, dry_run=dry_run
        )
        return
    deploy_plan.provision_command(
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
    pod_id: Annotated[str, typer.Argument(help="Pod id from `ta runpod up`")],
    timeout: Annotated[
        int, typer.Option("--timeout", help="Give up after this many seconds")
    ] = 900,
) -> None:
    """Block until a pod exposes SSH, then print `<ip> <port>`."""
    deploy_plan.wait_command(pod_id=pod_id, timeout_s=timeout)


@app.command()
def deploy(
    host: HostArg,
    port: PortArg,
    skip_setup: Annotated[
        bool, typer.Option("--skip-setup", help="Skip remote environment setup")
    ] = False,
    skip_sync: Annotated[bool, typer.Option("--skip-sync", help="Skip project file sync")] = False,
    skip_deps: Annotated[
        bool, typer.Option("--skip-deps", help="Skip Python dependency installation")
    ] = False,
) -> None:
    """Deploy ASR project to a RunPod instance."""
    conn = connect(host, port)

    project_root = get_project_root()

    if not skip_setup:
        setup_remote_environment(conn)

    if not skip_sync:
        sync_project(conn, project_root)

    if not skip_deps:
        install_dependencies(conn)

    print("\nDeployment finished!")
    print(f"To connect: ssh -i ~/.ssh/id_ed25519 -p {port} root@{host}")


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
# remote_scripts.script_preamble, so they are the same paths the training script will use.
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
    free = _remote_free_gib(conn)
    if free is None:
        print("Could not read `df /workspace`; skipping the disk preflight.")
        return
    try:
        need = deploy_plan.build_plan(experiment, overrides, 512).disk["recommended"]
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
    host: HostArg,
    port: PortArg,
    experiment: Annotated[
        str, typer.Option("--experiment", "-e", help=EXPERIMENT_HELP)
    ] = "granite_qwen_frozen",
    session_name: Annotated[
        str | None, typer.Option("--session-name", help=SESSION_NAME_HELP)
    ] = None,
    no_attach: Annotated[bool, typer.Option("--no-attach", help=NO_ATTACH_HELP)] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help=FORCE_HELP)] = False,
    skip_disk_check: Annotated[
        bool,
        typer.Option("--skip-disk-check", help="Start even if /workspace looks too small"),
    ] = False,
    wandb_run_id: Annotated[
        str | None,
        typer.Option("--wandb-run-id", envvar="WANDB_RUN_ID", help="W&B run ID to resume"),
    ] = None,
    wandb_resume: Annotated[
        str | None,
        typer.Option(
            "--wandb-resume", envvar="WANDB_RESUME", help="W&B resume mode: must, allow, or never"
        ),
    ] = None,
    hf_token: Annotated[
        str, typer.Option("--hf-token", envvar="HF_TOKEN", help=HF_TOKEN_HELP)
    ] = "",
    overrides: Annotated[list[str] | None, typer.Argument(help=OVERRIDES_HELP)] = None,
) -> None:
    """Start training on a remote RunPod instance in a tmux session."""
    conn = connect(host, port)

    overrides = list(overrides or [])
    if not skip_disk_check:
        _check_remote_disk(conn, experiment, overrides)

    session_name = _prepare_session(
        conn, session_name or _auto_session_name(f"train_{experiment}"), force, hf_token
    )

    print(f"\nStarting training session '{session_name}' with experiment '{experiment}'...")
    if overrides:
        print(f"Hydra overrides: {' '.join(overrides)}")

    _start_remote_tmux_script(
        conn,
        host,
        port,
        session_name,
        remote_scripts.build_training_script(
            experiment, hf_token, wandb_run_id, wandb_resume, overrides
        ),
        f"/tmp/train_{session_name}.sh",
        no_attach,
    )


@app.command()
def attach(
    host: HostArg,
    port: PortArg,
    session_name: Annotated[
        str | None,
        typer.Option("--session-name", help="tmux session name (default: prompt if several)"),
    ] = None,
    list_sessions: Annotated[
        bool, typer.Option("--list", "-l", help="List all sessions and exit")
    ] = False,
    logs: Annotated[
        bool, typer.Option("--logs", help="Show recent logs instead of attaching")
    ] = False,
    lines: Annotated[int, typer.Option("--lines", help="Number of log lines to show")] = 100,
) -> None:
    """Attach to, list, or view logs from a tmux session on a remote RunPod instance."""
    conn = connect(host, port)

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
            raise typer.Exit(1)
        if len(sessions) == 1:
            session_name = sessions[0]
            print(f"\nFound one active session: '{session_name}'. Proceeding automatically.")
        else:
            print("\nMultiple active sessions found. Please choose one:")
            session_name = Prompt.ask("Session", choices=sessions, default=sessions[0])

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


@app.command("eval")
def eval_model(
    host: HostArg,
    port: PortArg,
    model: Annotated[str, typer.Option("--model", "-m", help="Model path/ID or 'assemblyai'")],
    datasets: Annotated[
        list[str] | None,
        typer.Option(
            "--datasets", "-d", help="Datasets to evaluate on ('all' for every ASR dataset)"
        ),
    ] = None,
    max_samples: Annotated[
        int | None,
        typer.Option("--max-samples", "-n", help="Maximum samples to evaluate per dataset"),
    ] = None,
    assemblyai_model: Annotated[
        AssemblyAIModel, typer.Option("--assemblyai-model", help="AssemblyAI model")
    ] = AssemblyAIModel.universal_3_pro,
    num_workers: Annotated[
        int,
        typer.Option("--num-workers", "-w", help="Number of parallel workers for API evaluations"),
    ] = 1,
    streaming: Annotated[
        bool,
        typer.Option("--streaming", "-s", help="Use streaming evaluation (local or AssemblyAI)"),
    ] = False,
    session_name: Annotated[
        str | None, typer.Option("--session-name", help=SESSION_NAME_HELP)
    ] = None,
    no_attach: Annotated[bool, typer.Option("--no-attach", help=NO_ATTACH_HELP)] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help=FORCE_HELP)] = False,
    hf_token: Annotated[
        str, typer.Option("--hf-token", envvar="HF_TOKEN", help=HF_TOKEN_HELP)
    ] = "",
    assemblyai_api_key: Annotated[
        str,
        typer.Option(
            "--assemblyai-api-key",
            envvar="ASSEMBLYAI_API_KEY",
            help="AssemblyAI API key (required when --model assemblyai)",
        ),
    ] = "",
    extra_args: Annotated[
        list[str] | None,
        typer.Argument(help="Extra arguments passed through to `ta eval` on the pod"),
    ] = None,
) -> None:
    """Run ASR evaluation on a remote RunPod instance.

    Examples:
        ta runpod eval <HOST> <PORT> -m mazesmazes/tiny-audio -d loquacious
        ta runpod eval <HOST> <PORT> -m assemblyai --assemblyai-model universal -d loquacious -w 4
    """
    conn = connect(host, port)

    model_short = get_model_name(model)
    session_name = _prepare_session(
        conn, session_name or _auto_session_name(f"eval_{model_short}"), force, hf_token
    )
    if model == "assemblyai" and not assemblyai_api_key:
        msg = "set ASSEMBLYAI_API_KEY or pass --assemblyai-api-key when --model is assemblyai"
        raise typer.BadParameter(msg, param_hint="--assemblyai-api-key")

    if datasets is None:
        datasets = [DEFAULT_DATASET]

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
        remote_scripts.build_eval_script(
            hf_token,
            model,
            datasets,
            max_samples,
            assemblyai_api_key,
            assemblyai_model.value,
            num_workers,
            streaming,
            extra_args,
        ),
        f"/tmp/eval_{session_name}.sh",
        no_attach,
    )


@app.command()
def serve(
    host: HostArg,
    port: PortArg,
    model: Annotated[
        str, typer.Option("--model", "-m", help="HuggingFace Hub model ID or pod-local checkpoint")
    ] = "mazesmazes/tiny-audio",
    max_batch_size: Annotated[
        int | None,
        typer.Option("--max-batch-size", help="Most chunks per GPU batch (default: 32)"),
    ] = None,
    api_key: Annotated[
        str,
        typer.Option(
            "--api-key",
            envvar="TINY_AUDIO_API_KEY",
            help="Require this bearer key on requests (default: open server)",
        ),
    ] = "",
    session_name: Annotated[
        str | None, typer.Option("--session-name", help=SESSION_NAME_HELP)
    ] = None,
    no_attach: Annotated[bool, typer.Option("--no-attach", help=NO_ATTACH_HELP)] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help=FORCE_HELP)] = False,
    hf_token: Annotated[
        str, typer.Option("--hf-token", envvar="HF_TOKEN", help=HF_TOKEN_HELP)
    ] = "",
) -> None:
    """Start the batched inference server (`ta serve`) on a pod, behind its HTTP proxy.

    The pod must expose the server port over HTTP, which `ta runpod up --serve`
    does. Run `ta runpod deploy` first: it installs the fast kernels the server
    requires on CUDA.
    """
    conn = connect(host, port)
    session_name = _prepare_session(conn, session_name or "serve", force, hf_token)
    # SSH sessions don't inherit the pod's env; its PID 1 has RUNPOD_POD_ID.
    pod_id = conn.run(remote_scripts.POD_ID_COMMAND, hide=True, warn=True).stdout.strip()
    url = f"https://{pod_id}-{SERVE_PORT}.proxy.runpod.net" if pod_id else "(pod id unknown)"
    print(f"\nStarting server session '{session_name}' for {model}")
    print(f"URL once loaded: {url}   (GET /health answers when ready)")
    _start_remote_tmux_script(
        conn,
        host,
        port,
        session_name,
        remote_scripts.build_serve_script(hf_token, model, SERVE_PORT, max_batch_size, api_key),
        f"/tmp/serve_{session_name}.sh",
        no_attach,
    )


@app.command()
def checkpoint(host: HostArg, port: PortArg) -> None:
    """Find the latest checkpoint on a remote training server."""
    conn = connect(host, port)

    result = conn.run(
        "find /workspace/outputs -name 'checkpoint-*' -type d 2>/dev/null | sort -V | tail -1",
        hide=True,
        warn=True,
    )

    if not result.ok:
        typer.echo(f"Error: {result.stderr}", err=True)
        raise typer.Exit(1)

    ckpt = result.stdout.strip()
    if ckpt:
        print(ckpt)
    else:
        typer.echo("No checkpoints found", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
