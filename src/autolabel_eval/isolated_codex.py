from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .utils import write_json


FORBIDDEN_TRACE_MARKERS = (
    "/home/sangyu/Desktop/Master/codex_research_autolabeling/",
    "/home/sangyu/Desktop/Master/SpecLens/",
    "student_model_benchmark_results",
)

FORBIDDEN_EXEC_MARKERS = (
    "\nexec\n",
    "\nexec\r\n",
)


def _norm_ext(path: Path) -> str:
    ext = path.suffix.lower()
    return ext if ext else ".png"


def _find_forbidden_trace_hits(*chunks: str) -> list[str]:
    haystack = "\n".join(str(chunk or "") for chunk in chunks).lower()
    hits: list[str] = []
    for marker in FORBIDDEN_TRACE_MARKERS:
        if marker.lower() in haystack:
            hits.append(marker)
    return hits


def _find_forbidden_exec_hits(*chunks: str) -> list[str]:
    haystack = "\n".join(str(chunk or "") for chunk in chunks)
    hits: list[str] = []
    for marker in FORBIDDEN_EXEC_MARKERS:
        if marker in haystack:
            hits.append(marker.strip())
    return hits


def _prepend_isolation_guard(prompt_text: str) -> str:
    guard = (
        "Isolation rule:\n"
        "- Use only this prompt text and the attached staged images.\n"
        "- Do not run shell commands or inspect cwd, /tmp, repo files, logs, history, or any external paths.\n"
        "- Do not try to discover prior experiments, prior labels, or hidden context.\n"
        "- If the images are insufficient, answer from the visible evidence only rather than searching for more context.\n\n"
    )
    return guard + str(prompt_text)


def _parse_json_from_stdout(stdout_text: str) -> dict[str, Any]:
    text = str(stdout_text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    decoder = json.JSONDecoder()
    starts = [idx for idx, ch in enumerate(text) if ch == "{"][::-1]
    for start in starts:
        try:
            parsed, end = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if text[start + end :].strip():
            continue
        return parsed if isinstance(parsed, dict) else {}
    return {}


def run_isolated_codex_exec(
    *,
    artifact_dir: Path,
    artifact_stem: str,
    prompt_text: str,
    schema: dict[str, Any] | None,
    images: list[Path],
    model: str,
    reasoning_effort: str,
    temp_prefix: str = "codex_iso_",
    strict_trace_check: bool = True,
    timeout_sec: float | None = 300.0,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifact_dir / f"{artifact_stem}.stdout.txt"
    stderr_path = artifact_dir / f"{artifact_stem}.stderr.txt"
    prompt_copy_path = artifact_dir / f"{artifact_stem}.prompt.md"
    output_copy_path = artifact_dir / f"{artifact_stem}.json"

    guarded_prompt_text = _prepend_isolation_guard(prompt_text)
    prompt_copy_path.write_text(guarded_prompt_text)
    if schema is not None:
        schema_copy_path = artifact_dir / f"{artifact_stem}.schema.json"
        write_json(schema_copy_path, schema)

    with tempfile.TemporaryDirectory(prefix=temp_prefix, dir="/tmp") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        prompt_path = tmpdir / "prompt.md"
        out_json = tmpdir / "output.json"
        prompt_path.write_text(guarded_prompt_text)
        if schema is not None:
            schema_path = tmpdir / "output_schema.json"
            write_json(schema_path, schema)

        staged_inputs: list[dict[str, str]] = []
        cmd = [
            "codex",
            "exec",
            "-c",
            f'model="{model}"',
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            "-s",
            "workspace-write",
            "--skip-git-repo-check",
        ]
        if schema is not None:
            cmd.extend(
                [
                    "--output-schema",
                    str(schema_path.name),
                    "-o",
                    str(out_json.name),
                ]
            )
        for idx, image in enumerate(images):
            source_path = Path(image).resolve()
            staged_name = f"example_{idx:02d}{_norm_ext(source_path)}"
            staged_path = tmpdir / staged_name
            shutil.copy2(source_path, staged_path)
            staged_inputs.append(
                {
                    "rank": idx,
                    "source_path": str(source_path),
                    "staged_name": staged_name,
                }
            )
            cmd.extend(["-i", staged_name])
        cmd.append("-")

        start = time.time()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(tmpdir),
                capture_output=True,
                text=True,
                input=guarded_prompt_text,
                timeout=float(timeout_sec) if timeout_sec is not None else None,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            proc = subprocess.CompletedProcess(
                args=cmd,
                returncode=124,
                stdout=str(exc.stdout or ""),
                stderr=((str(exc.stderr or "") + "\n[isolated_codex] timed out") if exc.stderr else "[isolated_codex] timed out"),
            )
        elapsed = time.time() - start

        stdout_path.write_text(proc.stdout)
        stderr_path.write_text(proc.stderr)

        output: dict[str, Any] = {}
        if schema is not None and out_json.exists():
            shutil.copy2(out_json, output_copy_path)
            try:
                output = json.loads(out_json.read_text())
            except Exception:
                output = {}
        elif schema is None:
            output = _parse_json_from_stdout(proc.stdout)
            if output:
                write_json(output_copy_path, output)

    forbidden_trace_hits = _find_forbidden_trace_hits(proc.stdout, proc.stderr)
    forbidden_exec_hits = _find_forbidden_exec_hits(proc.stdout, proc.stderr)
    if strict_trace_check and (forbidden_trace_hits or forbidden_exec_hits):
        all_hits = list(forbidden_trace_hits) + list(forbidden_exec_hits)
        raise RuntimeError(
            f"Forbidden isolation markers detected for {artifact_stem}: {all_hits}"
        )

    return {
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "elapsed_sec": elapsed,
        "output": output,
        "staged_inputs": staged_inputs,
        "forbidden_trace_hits": list(forbidden_trace_hits) + list(forbidden_exec_hits),
        "timed_out": bool(timed_out),
    }
