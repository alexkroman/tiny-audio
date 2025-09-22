#!/usr/bin/env python3
"""Attach to an existing tmux session on remote RunPod instance."""

import argparse
import subprocess
import sys


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command."""
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, check=check)
    if capture_output:
        return result.stdout.strip()
    return result.returncode


def check_ssh_connection(host, port):
    """Test SSH connection to the RunPod instance."""
    print(f"Testing SSH connection to {host}:{port}...")
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@{host} 'echo Connected'"
    try:
        run_command(cmd, capture_output=True)
        print("SSH connection successful!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to connect via SSH. Please check:")
        print("  - The host and port are correct")
        print("  - The pod is running")
        print("  - Your SSH key ~/.ssh/id_ed25519 is added to the pod")
        return False


def list_sessions(host, port):
    """List all tmux sessions on the remote instance."""
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} 'tmux list-sessions 2>/dev/null || echo NO_SESSIONS'"
    try:
        result = run_command(cmd, capture_output=True)
        if "NO_SESSIONS" in result or not result:
            return []

        sessions = []
        for line in result.split("\n"):
            if line:
                # Parse tmux list-sessions output (format: "session_name: ...")
                session_name = line.split(":")[0].strip()
                sessions.append(session_name)
        return sessions
    except subprocess.CalledProcessError:
        return []


def get_session_logs(host, port, session_name, lines=100):
    """Get the last N lines from a tmux session."""
    print(f"\nGetting last {lines} lines from session '{session_name}'...")
    cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        'tmux capture-pane -t {session_name} -p -S -{lines} 2>/dev/null || echo "Session not found"'"""

    try:
        result = run_command(cmd, capture_output=True)
        if "Session not found" in result:
            print(f"Session '{session_name}' not found")
            return None
        return result
    except subprocess.CalledProcessError:
        return None


def attach_to_session(host, port, session_name):
    """Attach to a tmux session on the remote instance."""
    print(f"\nAttaching to session '{session_name}'...")
    print("=" * 50)
    print("TMUX CONTROLS:")
    print("  - Detach (leave running): Ctrl+B then D")
    print("  - Scroll up/down: Ctrl+B then [ (then q to exit scroll mode)")
    print("  - Kill session: Ctrl+B then : then type 'kill-session'")
    print("  - Switch panes: Ctrl+B then arrow keys")
    print("=" * 50)
    print("")

    # SSH and attach to tmux session
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no -t root@{host} 'tmux attach-session -t {session_name}'"

    # Use subprocess.call for interactive session
    result = subprocess.call(cmd, shell=True)

    if result != 0:
        print(f"\nFailed to attach to session '{session_name}'")
        print("The session may not exist or may have ended.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Attach to tmux session on remote RunPod instance")
    parser.add_argument("host", help="RunPod instance IP address or hostname")
    parser.add_argument("port", type=int, help="SSH port for the RunPod instance")
    parser.add_argument(
        "--session-name", default=None, help="Name of the tmux session to attach to"
    )
    parser.add_argument("--list", action="store_true", help="List all available tmux sessions")
    parser.add_argument(
        "--logs", action="store_true", help="Show recent logs from session without attaching"
    )
    parser.add_argument(
        "--lines", type=int, default=100, help="Number of log lines to show (default: 100)"
    )

    args = parser.parse_args()

    # Test SSH connection
    if not check_ssh_connection(args.host, args.port):
        sys.exit(1)

    # List sessions if requested
    if args.list:
        print("\nAvailable tmux sessions:")
        sessions = list_sessions(args.host, args.port)
        if not sessions:
            print("  No active sessions found")
        else:
            for session in sessions:
                print(f"  - {session}")

        if not args.session_name and sessions:
            print("\nTo attach to a session, run:")
            print(
                f"  python scripts/attach_remote_session.py {args.host} {args.port} --session-name <session_name>"
            )
        return

    # If no session name provided, try to find one
    if not args.session_name:
        sessions = list_sessions(args.host, args.port)
        if not sessions:
            print("\nNo active tmux sessions found on the remote instance.")
            print("Start a new training session with:")
            print(f"  python scripts/start_remote_training.py {args.host} {args.port}")
            sys.exit(1)
        elif len(sessions) == 1:
            args.session_name = sessions[0]
            print(f"\nFound one active session: '{args.session_name}'")
        else:
            print("\nMultiple active sessions found:")
            for i, session in enumerate(sessions, 1):
                print(f"  {i}. {session}")

            try:
                choice = input("\nSelect session number (or press Ctrl+C to cancel): ")
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    args.session_name = sessions[idx]
                else:
                    print("Invalid selection")
                    sys.exit(1)
            except (KeyboardInterrupt, ValueError):
                print("\nCancelled")
                sys.exit(1)

    # Show logs if requested
    if args.logs:
        logs = get_session_logs(args.host, args.port, args.session_name, args.lines)
        if logs:
            print("\n" + "=" * 50)
            print(f"Recent output from session '{args.session_name}':")
            print("=" * 50)
            print(logs)
            print("=" * 50)
            print("\nTo attach to this session, run:")
            print(
                f"  python scripts/attach_remote_session.py {args.host} {args.port} --session-name {args.session_name}"
            )
        else:
            print(f"Could not get logs from session '{args.session_name}'")
        return

    # Attach to the session
    if not attach_to_session(args.host, args.port, args.session_name):
        # If attachment failed, list available sessions
        print("\nAvailable sessions:")
        sessions = list_sessions(args.host, args.port)
        if not sessions:
            print("  No active sessions found")
        else:
            for session in sessions:
                print(f"  - {session}")


if __name__ == "__main__":
    main()
