#!/usr/bin/env python3
"""
Attach to, list, or view logs from an existing tmux session on a remote RunPod instance.
"""

import argparse
import subprocess
import sys


def run_command(cmd, check=True, capture_output=False):
    """
    Executes a shell command.

    Args:
        cmd (str): The command to run.
        check (bool): If True, raises CalledProcessError on non-zero exit codes.
        capture_output (bool): If True, captures and returns stdout.

    Returns:
        Union[str, int]: The stripped stdout if capture_output is True, otherwise the return code.
    """
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, check=check)
    if capture_output:
        return result.stdout.strip()
    return result.returncode


def check_ssh_connection(host, port):
    """
    Tests the SSH connection to the remote host.

    Args:
        host (str): The IP address or hostname of the remote machine.
        port (int): The SSH port.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    print(f"📡 Testing SSH connection to {host}:{port}...")
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@{host} 'echo Connected'"
    try:
        run_command(cmd, capture_output=True)
        print("✅ SSH connection successful!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to connect via SSH. Please check:")
        print(f"  - The host ({host}) and port ({port}) are correct.")
        print("  - The pod is running.")
        print("  - Your SSH key (~/.ssh/id_ed25519) is authorized on the pod.")
        return False


def list_sessions(host, port):
    """
    Fetches a list of all running tmux sessions on the remote host.

    Args:
        host (str): The IP address or hostname of the remote machine.
        port (int): The SSH port.

    Returns:
        list[str]: A list of tmux session names.
    """
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} 'tmux list-sessions -F \"#S\" 2>/dev/null'"
    try:
        result = run_command(cmd, capture_output=True)
        return result.split("\n") if result else []
    except subprocess.CalledProcessError:
        return []


def get_session_logs(host, port, session_name, lines=100):
    """
    Captures and returns the last N lines of output from a tmux pane.

    Args:
        host (str): The IP address or hostname of the remote machine.
        port (int): The SSH port.
        session_name (str): The name of the target tmux session.
        lines (int): The number of recent lines to capture.

    Returns:
        Optional[str]: The captured logs as a string, or None if the session is not found.
    """
    print(f"\n📄 Getting last {lines} lines from session '{session_name}'...")
    cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        "tmux capture-pane -t '{session_name}' -p -S -{lines}"
    """
    try:
        return run_command(cmd, capture_output=True)
    except subprocess.CalledProcessError:
        print(f"❌ Session '{session_name}' not found or an error occurred.")
        return None


def attach_to_session(host, port, session_name):
    """
    Attaches to a specific tmux session in an interactive terminal.

    Args:
        host (str): The IP address or hostname of the remote machine.
        port (int): The SSH port.
        session_name (str): The name of the tmux session to attach to.
    """
    print(f"\n🖥️  Attaching to session '{session_name}'...")
    print("=" * 50)
    print("TMUX CONTROLS:")
    print("  - Detach (and leave running): Ctrl+B then D")
    print("  - Scroll Mode:              Ctrl+B then [ (use arrows, q to exit)")
    print("=" * 50)

    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no -t root@{host} \"tmux attach-session -t '{session_name}'\""
    subprocess.run(cmd, shell=True)
    print(f"\nDetached from session '{session_name}'.")


def main():
    """Main function to handle argument parsing and session management."""
    parser = argparse.ArgumentParser(
        description="Attach to, list, or view logs from a tmux session on a remote RunPod instance.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("host", help="RunPod instance IP address or hostname.")
    parser.add_argument("port", type=int, help="SSH port for the RunPod instance.")
    parser.add_argument("--session-name", help="Name of the tmux session to interact with.")
    parser.add_argument(
        "--list", action="store_true", help="List all available tmux sessions and exit."
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Show recent logs from the session instead of attaching.",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=100,
        help="Number of log lines to show with --logs (default: 100).",
    )

    args = parser.parse_args()

    if not check_ssh_connection(args.host, args.port):
        sys.exit(1)

    sessions = list_sessions(args.host, args.port)

    if args.list:
        print("\nAvailable tmux sessions:")
        if not sessions:
            print("  No active sessions found.")
        else:
            for session in sessions:
                print(f"  - {session}")
        return

    if not args.session_name:
        if not sessions:
            print("\nNo active tmux sessions found. Start a new training session first.")
            sys.exit(1)
        elif len(sessions) == 1:
            args.session_name = sessions[0]
            print(f"\nFound one active session: '{args.session_name}'. Proceeding automatically.")
        else:
            print("\nMultiple active sessions found. Please choose one:")
            for i, session in enumerate(sessions, 1):
                print(f"  {i}. {session}")
            try:
                choice = input(f"Enter number (1-{len(sessions)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    args.session_name = sessions[idx]
                else:
                    print("Invalid selection. Exiting.")
                    sys.exit(1)
            except (KeyboardInterrupt, ValueError):
                print("\nSelection cancelled. Exiting.")
                sys.exit(0)

    if args.logs:
        logs = get_session_logs(args.host, args.port, args.session_name, args.lines)
        if logs:
            print("\n" + "=" * 50)
            print(logs)
            print("=" * 50)
    else:
        attach_to_session(args.host, args.port, args.session_name)


if __name__ == "__main__":
    main()
