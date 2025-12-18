#!/usr/bin/env python3
"""
Shared utilities for remote RunPod operations using Fabric.
"""

from pathlib import Path

from fabric import Connection
from invoke import UnexpectedExit

SSH_KEY_PATH = "~/.ssh/id_ed25519"


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
        print("SSH connection successful!")
        return True
    except Exception as e:
        print(f"Failed to connect via SSH: {e}")
        return False


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
    import subprocess

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
    subprocess.run(cmd, shell=True)
    print(f"\nDetached from session '{session_name}'.")
