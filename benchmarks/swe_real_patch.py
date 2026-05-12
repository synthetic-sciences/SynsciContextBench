"""Real-patch SWE evaluation mode.

The default SWE-Agent benchmark scores standalone code with structural checks
(``contains_return``, ``contains_try_except`` ...). The diagnosis flagged this
as too forgiving: an LLM can produce plausible code that satisfies every
structural check and still not fix the bug. This module adds an opt-in
*real-patch* path: when a test case has both ``repo_url`` and ``test_command``
fields, we

1. Clone or refresh a shallow copy of the repo into a per-case sandbox.
2. Apply the LLM's generated patch (or write the standalone solution to the
   referenced target file).
3. Run the test command in the sandbox.
4. Score success on exit code + (optional) test-name pass rate.

The module is intentionally cautious: it skips silently when ``repo_url`` is
missing, refuses to run if the test command looks dangerous, and isolates
sandboxes in a temp dir that is cleaned up after each case.

This is invoked by the SWE-Agent benchmark when ``--real-patch`` is on; it
augments — not replaces — the existing judge composite, so we get *both*
signals in the report.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


# Patterns that cannot appear in a sandboxed test command. Better to refuse a
# case than to run arbitrary shell.
_FORBIDDEN_PATTERNS = (
    re.compile(r"\brm\s+-rf?\s+/"),
    re.compile(r"\bsudo\b"),
    re.compile(r"curl\s+[^|]*\|\s*(?:sh|bash)"),
    re.compile(r"\bsystemctl\b"),
    re.compile(r"\bdd\b"),
    # fork-bomb sketch (we just reject anything that looks like ":(){...};:")
    re.compile(r":\(\)\s*\{[^}]+\}\s*;:"),
)

_DANGEROUS_GIT_OPS = ("push", "force-push", "--force")


@dataclass
class RealPatchOutcome:
    ran: bool = False             # False if we skipped (missing fields, etc.)
    success: bool = False         # exit code 0 from test command
    exit_code: int = -1
    test_pass_rate: float = 0.0   # parsed from pytest output, fallback 0
    runtime_ms: float = 0.0
    skipped_reason: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


def _is_safe_command(cmd: str) -> bool:
    """Defense-in-depth: refuse anything that looks like it can damage the host."""
    for pat in _FORBIDDEN_PATTERNS:
        if pat.search(cmd):
            return False
    tokens = shlex.split(cmd)
    for danger in _DANGEROUS_GIT_OPS:
        if danger in tokens:
            return False
    return True


def _parse_pytest_pass_rate(stdout: str) -> float:
    """Parse a pytest-ish summary line. Returns 0 if unparseable."""
    if not stdout:
        return 0.0
    # pytest: "5 passed, 2 failed in 1.23s" or "1 failed, 4 passed"
    passed = re.search(r"(\d+)\s+passed", stdout)
    failed = re.search(r"(\d+)\s+failed", stdout)
    if passed or failed:
        p = int(passed.group(1)) if passed else 0
        f = int(failed.group(1)) if failed else 0
        total = p + f
        return p / total if total else 0.0
    return 0.0


def _shallow_clone(repo_url: str, dest: Path) -> None:
    """Shallow git clone into `dest`. Raises CalledProcessError on failure."""
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True, capture_output=True,
    )


def _looks_like_diff(text: str) -> bool:
    """Heuristic: is the generated solution a unified diff/patch?"""
    head = (text or "")[:1000]
    return ("--- a/" in head and "+++ b/" in head) or head.startswith("diff --git")


def _apply_patch(sandbox: Path, patch_text: str) -> bool:
    """Apply a unified diff to `sandbox`. Returns True on success."""
    if not _looks_like_diff(patch_text):
        return False
    try:
        proc = subprocess.run(
            ["git", "apply", "--whitespace=fix", "-"],
            cwd=str(sandbox), input=patch_text, text=True,
            capture_output=True, check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _write_standalone(sandbox: Path, target_rel: str, content: str) -> Path:
    """Overwrite (or create) a target file with the standalone solution."""
    target = sandbox / target_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


async def run_real_patch(
    *,
    repo_url: str,
    test_command: str,
    solution: str,
    target_file: str = "",
    timeout_s: int = 120,
) -> RealPatchOutcome:
    """Run the real-patch eval for one solution.

    The eval will:
      1. Refuse the test command if it looks dangerous.
      2. Shallow-clone the repo into a temp dir.
      3. Apply the solution as a patch if it looks like a diff, otherwise
         overwrite ``target_file`` with the standalone solution.
      4. Run ``test_command`` with a hard timeout.
    """
    if not repo_url or not test_command:
        return RealPatchOutcome(skipped_reason="missing repo_url or test_command")
    if not _is_safe_command(test_command):
        return RealPatchOutcome(skipped_reason="test_command rejected by safety filter")

    sandbox = Path(tempfile.mkdtemp(prefix="swe-real-"))
    t0 = time.perf_counter()
    outcome = RealPatchOutcome(ran=True)
    try:
        try:
            _shallow_clone(repo_url, sandbox)
        except subprocess.CalledProcessError as e:
            outcome.skipped_reason = f"clone failed: {e.stderr.decode()[:200]}"
            outcome.ran = False
            return outcome

        if _looks_like_diff(solution):
            applied = _apply_patch(sandbox, solution)
            if not applied:
                outcome.skipped_reason = "patch failed to apply"
                return outcome
        elif target_file:
            _write_standalone(sandbox, target_file, solution)
        else:
            outcome.skipped_reason = "standalone solution but no target_file"
            return outcome

        # Run the test command with a wall-clock cap.
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_shell(
                    test_command,
                    cwd=str(sandbox),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                ),
                timeout=timeout_s,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
            outcome.exit_code = proc.returncode if proc.returncode is not None else -1
            outcome.success = outcome.exit_code == 0
            outcome.stdout_tail = stdout_bytes.decode(errors="replace")[-2000:]
            outcome.stderr_tail = stderr_bytes.decode(errors="replace")[-2000:]
            outcome.test_pass_rate = _parse_pytest_pass_rate(outcome.stdout_tail)
            if outcome.test_pass_rate == 0.0 and outcome.success:
                # No pytest output parsed but exit-code zero — treat as full pass.
                outcome.test_pass_rate = 1.0
        except asyncio.TimeoutError:
            outcome.skipped_reason = f"test command timed out after {timeout_s}s"
            return outcome
    finally:
        outcome.runtime_ms = (time.perf_counter() - t0) * 1000
        shutil.rmtree(sandbox, ignore_errors=True)

    return outcome
