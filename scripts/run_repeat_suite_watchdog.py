from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    return env


def _list_processes() -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["ps", "-eo", "pid,args"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines()[1:]:
        text = line.strip()
        if not text:
            continue
        parts = text.split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, args_text = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        rows.append({"pid": pid, "args": args_text})
    return rows


def _find_matching_suite_pids(*, session_prefix: str, process_substring: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _list_processes():
        args_text = str(row["args"])
        if "run_repeat_suite_watchdog.py" in args_text:
            continue
        if str(process_substring) not in args_text:
            continue
        if str(session_prefix) not in args_text:
            continue
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-prefix", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--duration-hours", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--process-substring", default="run_cyancross_shortdesc_repeat_suite.py")
    parser.add_argument("--log-path", default="")
    parser.add_argument("--status-path", default="")
    args, extra = parser.parse_known_args()

    if extra and extra[0] == "--":
        extra = extra[1:]
    if not extra:
        raise SystemExit("Pass the exact repeat-suite command after `--`.")

    workspace_root = Path(args.workspace_root).resolve()
    suite_root = workspace_root / "outputs" / "repeat_suites" / str(args.session_prefix)
    suite_root.mkdir(parents=True, exist_ok=True)
    aggregate_summary = suite_root / "aggregate_summary.json"
    log_path = Path(args.log_path).resolve() if str(args.log_path).strip() else suite_root / "watchdog.log"
    status_path = Path(args.status_path).resolve() if str(args.status_path).strip() else suite_root / "watchdog_status.json"

    deadline = time.time() + float(args.duration_hours) * 3600.0
    suite_cmd = list(extra)
    last_spawn_pid: int | None = None

    while True:
        now = time.time()
        matching = _find_matching_suite_pids(
            session_prefix=str(args.session_prefix),
            process_substring=str(args.process_substring),
        )
        completed = aggregate_summary.exists()
        timed_out = now >= deadline

        state = "watching"
        if completed:
            state = "completed"
        elif timed_out:
            state = "timed_out"
        elif not matching:
            state = "launching"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{_now_iso()}] launching suite: {shlex.join(suite_cmd)}\n")
                handle.flush()
                proc = subprocess.Popen(
                    suite_cmd,
                    cwd=str(ROOT),
                    stdout=handle,
                    stderr=handle,
                    text=True,
                    start_new_session=True,
                    env=_subprocess_env(),
                )
                last_spawn_pid = int(proc.pid)
            matching = _find_matching_suite_pids(
                session_prefix=str(args.session_prefix),
                process_substring=str(args.process_substring),
            )

        status_payload = {
            "session_prefix": str(args.session_prefix),
            "workspace_root": str(workspace_root),
            "suite_root": str(suite_root),
            "aggregate_summary_json": str(aggregate_summary),
            "state": state,
            "timestamp": _now_iso(),
            "deadline_unix": float(deadline),
            "deadline_iso": datetime.fromtimestamp(deadline, tz=timezone.utc).astimezone().isoformat(timespec="seconds"),
            "poll_seconds": float(args.poll_seconds),
            "process_substring": str(args.process_substring),
            "matching_suite_pids": matching,
            "last_spawn_pid": last_spawn_pid,
            "suite_cmd": suite_cmd,
            "completed": bool(completed),
            "timed_out": bool(timed_out),
            "watchdog_log": str(log_path),
        }
        _write_json(status_path, status_payload)

        if completed:
            return
        if timed_out:
            raise SystemExit(1)
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    main()
