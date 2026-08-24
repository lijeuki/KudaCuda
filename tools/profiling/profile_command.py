"""Generate or run Nsight Systems/Compute profiling commands."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def build_profile_command(
    tool: str,
    command: Sequence[str],
    output: str,
    *,
    executable: str | None = None,
) -> list[str]:
    profiler = executable or tool
    if tool == "nsys":
        return [profiler, "profile", "--force-overwrite=true", "-o", output, *command]
    return [profiler, "--set", "full", "-o", output, *command]


def resolve_profiler(tool: str) -> str | None:
    """Resolve PATH tools and standard NVIDIA Windows installations."""

    resolved = shutil.which(tool)
    if resolved is not None:
        return resolved
    if os.name != "nt":
        return None

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    nvidia_root = program_files / "NVIDIA Corporation"
    patterns = {
        "nsys": "Nsight Systems */target-windows-x64/nsys.exe",
        "ncu": "Nsight Compute */target/windows-desktop-win7-x64/ncu.exe",
    }
    matches = sorted(nvidia_root.glob(patterns[tool]))
    return str(matches[-1]) if matches else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=("nsys", "ncu"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("a command is required after --")
    executable = resolve_profiler(args.tool)
    if executable is None:
        payload = {"status": "skipped", "reason": f"{args.tool} not found", "command": command}
        print(json.dumps(payload, sort_keys=True))
        return 2
    profile_command = build_profile_command(args.tool, command, args.output, executable=executable)
    payload = {"status": "ready", "tool": args.tool, "executable": executable, "command": profile_command}
    if args.run:
        completed = subprocess.run(profile_command, check=False)
        payload["status"] = "validated" if completed.returncode == 0 else "failed"
        payload["returncode"] = completed.returncode
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] in {"ready", "validated"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
