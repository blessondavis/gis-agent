"""Per-turn checkpoints of a job's artefacts, so a rewind restores the files.

Rewinding a conversation is easy; rewinding what the tools *did* is the part
agent harnesses usually skip. Grok Build's ``/rewind`` truncates the transcript
but leaves files as they are, and says so. Here a job's state is a handful of
files, so a checkpoint can restore them too.

Two kinds of artefact, two strategies:

* small ones (the manifest, mask, centrelines, edits) are copied when the turn
  starts -- a couple of megabytes;
* large ones (the per-chip confidence maps, tens of MB for a region) are only
  copied if and when the turn is about to overwrite them. Most turns never
  re-run the model, so most checkpoints stay small.

The MCP server process does the overwriting, so the "active checkpoint" is a
file in the job directory rather than state in either process's memory.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = ("job.json", "mask.tif", "roads.geojson", "network.geojson",
            "edits.json", "plan.json")
KEEP = 12


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Checkpoints:
    def __init__(self, job_dir: Path | str) -> None:
        self.job_dir = Path(job_dir)
        self.root = self.job_dir / "checkpoints"
        self.active_file = self.root / "ACTIVE"

    # -- lifecycle ---------------------------------------------------------- #

    def begin(self, label: str, message_index: int, **extra) -> str:
        """Snapshot the small artefacts and make this the active checkpoint."""
        self.root.mkdir(parents=True, exist_ok=True)
        existing = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        cid = f"{len(existing) + 1:04d}"
        while (self.root / cid).exists():
            cid = f"{int(cid) + 1:04d}"
        d = self.root / cid
        (d / "files").mkdir(parents=True)

        absent = []
        for name in SNAPSHOT:
            src = self.job_dir / name
            if src.exists():
                shutil.copy2(src, d / "files" / name)
            else:
                absent.append(name)
        meta = {"id": cid, "label": label[:160], "at": _utc(),
                "message_index": message_index, "absent": absent, "preserved": [],
                **extra}
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        self.active_file.write_text(cid, encoding="utf-8")
        self._prune()
        return cid

    def end(self) -> None:
        self.active_file.unlink(missing_ok=True)

    def active(self) -> str | None:
        try:
            cid = self.active_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return cid if (self.root / cid).is_dir() else None

    # -- copy on write ------------------------------------------------------ #

    def preserve(self, path: Path | str) -> None:
        """Call before overwriting a large artefact. Cheap no-op when inactive."""
        cid = self.active()
        if cid is None:
            return
        path = Path(path)
        try:
            rel = path.resolve().relative_to(self.job_dir.resolve()).as_posix()
        except ValueError:
            return
        d = self.root / cid
        meta = self._meta(cid)
        if rel in meta["preserved"] or rel in meta["absent"]:
            return                       # first write in this turn wins
        if path.exists():
            dst = d / "files" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            meta["preserved"].append(rel)
        else:
            meta["absent"].append(rel)
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    # -- query / restore ---------------------------------------------------- #

    def _meta(self, cid: str) -> dict:
        return json.loads((self.root / cid / "meta.json").read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        out = []
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            try:
                m = self._meta(d.name)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            out.append({k: m[k] for k in ("id", "label", "at", "message_index")}
                       | {"n_files": len(m["preserved"]) + len(SNAPSHOT) - len(
                           [a for a in m["absent"] if a in SNAPSHOT])})
        return out

    def restore(self, cid: str) -> dict:
        """Put every artefact back as it was when checkpoint ``cid`` began.

        Later checkpoints are discarded: they describe a future that no
        longer happened. Returns the checkpoint's metadata.
        """
        d = self.root / cid
        if not d.is_dir():
            raise FileNotFoundError(f"no such checkpoint: {cid}")
        # Changes made after ``cid`` may have been preserved by *later*
        # checkpoints instead; restoring newest-first and ending with ``cid``
        # leaves each file at its earliest recorded state.
        later = sorted(p.name for p in self.root.iterdir()
                       if p.is_dir() and p.name > cid)
        for other in reversed(later):
            self._apply(other, snapshot=False)
        meta = self._apply(cid, snapshot=True)
        for other in later:
            shutil.rmtree(self.root / other, ignore_errors=True)
        self.end()
        return meta

    def _apply(self, cid: str, *, snapshot: bool) -> dict:
        d = self.root / cid
        meta = self._meta(cid)
        names = list(meta["preserved"]) + (list(SNAPSHOT) if snapshot else [])
        for rel in names:
            src = d / "files" / rel
            if src.exists():
                dst = self.job_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        for rel in meta["absent"]:
            if snapshot or rel not in SNAPSHOT:
                (self.job_dir / rel).unlink(missing_ok=True)
        return meta

    def _prune(self) -> None:
        dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        for d in dirs[:-KEEP]:
            shutil.rmtree(d, ignore_errors=True)
