"""Headless QGIS via the qgis_process CLI.

Deliberately *not* the QGIS-plugin-over-a-socket approach that the popular
qgis_mcp servers use: those need a QGIS Desktop GUI running with a human
clicking "Start Server", which cannot be containerised or driven from a web
backend. qgis_process is the supported headless entry point, ships with every
QGIS install, and is what the official qgis/qgis Docker image exposes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from gisagent.config import get_settings


class QgisNotAvailable(RuntimeError):
    pass


class QgisAlgorithmError(RuntimeError):
    def __init__(self, algorithm: str, returncode: int, stderr: str, stdout: str = ""):
        self.algorithm = algorithm
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout
        super().__init__(
            f"{algorithm} failed (exit {returncode}): {stderr.strip()[:600]}"
        )


@dataclass
class QgisResult:
    algorithm: str
    ok: bool
    returncode: int
    outputs: dict = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


class QgisProcess:
    """Thin, typed wrapper around `qgis_process run`."""

    def __init__(self, exe: str | None = None) -> None:
        self._exe = exe or get_settings().qgis_process
        if not self._exe:
            raise QgisNotAvailable(
                "qgis_process not found. Install QGIS, or set GISAGENT_QGIS_PROCESS "
                "to the full path of qgis_process(.bat)."
            )

    @property
    def exe(self) -> str:
        return self._exe

    def _argv(self, args: list[str]) -> list[str]:
        # A .bat is not directly executable via CreateProcess in every context,
        # so route Windows shims through cmd.exe explicitly.
        if os.name == "nt" and self._exe.lower().endswith(".bat"):
            return ["cmd", "/c", self._exe, *args]
        return [self._exe, *args]

    def _run(self, args: list[str], timeout: float) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        # keep QGIS from trying to touch a user profile / display
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        return subprocess.run(
            self._argv(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )

    def version(self, timeout: float = 120) -> str:
        proc = self._run(["--version"], timeout)
        first = (proc.stdout or proc.stderr).strip().splitlines()
        return first[0] if first else "unknown"

    def list_algorithms(self, timeout: float = 240) -> dict[str, str]:
        """Return {algorithm_id: human name}."""
        proc = self._run(["list"], timeout)
        algs: dict[str, str] = {}
        for line in (proc.stdout or "").splitlines():
            m = re.match(r"^\s+([a-z0-9_]+:[A-Za-z0-9_.]+)\s+(.*)$", line)
            if m:
                algs[m.group(1)] = m.group(2).strip()
        return algs

    def run(
        self,
        algorithm: str,
        params: dict[str, object],
        *,
        timeout: float = 900,
        check: bool = True,
    ) -> QgisResult:
        """Run one processing algorithm.

        Values are stringified; Path becomes str, bool becomes true/false, and
        None is dropped so callers can pass optional params unconditionally.
        """
        import time

        args = ["run", algorithm, "--json"]
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, Path):
                value = str(value)
            args.append(f"--{key}={value}")

        t0 = time.perf_counter()
        proc = self._run(args, timeout)
        duration = time.perf_counter() - t0

        outputs: dict = {}
        stdout = proc.stdout or ""
        # qgis_process prints progress noise before the JSON document
        brace = stdout.find("{")
        if brace >= 0:
            try:
                parsed = json.loads(stdout[brace:])
                outputs = parsed.get("results", parsed)
            except json.JSONDecodeError:
                outputs = {}

        ok = proc.returncode == 0
        result = QgisResult(
            algorithm=algorithm,
            ok=ok,
            returncode=proc.returncode,
            outputs=outputs,
            stdout=stdout,
            stderr=proc.stderr or "",
            duration_s=duration,
        )
        if check and not ok:
            raise QgisAlgorithmError(algorithm, proc.returncode, result.stderr, stdout)
        return result


def available() -> bool:
    try:
        QgisProcess()
        return True
    except QgisNotAvailable:
        return False
