"""Local JSON / JSONL storage for runs.

Layout:  workspace/runs/<run_id>/
    manifest.json          run configuration (no secrets) and status
    suite.json             exact copy of the suite used
    target_records.jsonl   one line per target call, appended + fsynced as it completes
    comparisons.jsonl      one line per pairwise comparison, appended as it completes

Append-only JSONL means an interrupted run keeps every completed result. When a
record is retried on resume, the newest line for that id wins.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Optional

from . import config
from .schemas import BiasSuite, ComparisonRecord, RunManifest, TargetRecord, utcnow


class RunStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or config.RUNS_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    # paths --------------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            raise ValueError("invalid run id")
        return self.root / run_id

    def _p(self, run_id: str, name: str) -> Path:
        return self.run_dir(run_id) / name

    # manifest / suite ------------------------------------------------------------
    def create(self, manifest: RunManifest, suite: BiasSuite) -> None:
        d = self.run_dir(manifest.run_id)
        d.mkdir(parents=True, exist_ok=False)
        self._p(manifest.run_id, "suite.json").write_text(suite.model_dump_json(indent=2), encoding="utf-8")
        self.save_manifest(manifest)

    def save_manifest(self, manifest: RunManifest) -> None:
        manifest.updated_at = utcnow()
        path = self._p(manifest.run_id, "manifest.json")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def load_manifest(self, run_id: str) -> RunManifest:
        return RunManifest.model_validate_json(self._p(run_id, "manifest.json").read_text(encoding="utf-8"))

    def load_suite(self, run_id: str) -> BiasSuite:
        return BiasSuite.model_validate_json(self._p(run_id, "suite.json").read_text(encoding="utf-8"))

    # jsonl --------------------------------------------------------------------
    def _append(self, run_id: str, name: str, line: str) -> None:
        with open(self._p(run_id, name), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _iter(self, run_id: str, name: str) -> Iterator[dict]:
        path = self._p(run_id, name)
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line from a hard crash; skip it rather than lose the run.
                    continue

    def append_target(self, rec: TargetRecord) -> None:
        self._append(rec.run_id, "target_records.jsonl", rec.model_dump_json())

    def append_comparison(self, rec: ComparisonRecord) -> None:
        self._append(rec.run_id, "comparisons.jsonl", rec.model_dump_json())

    def load_targets(self, run_id: str) -> dict[str, TargetRecord]:
        out: dict[str, TargetRecord] = {}
        for d in self._iter(run_id, "target_records.jsonl"):
            rec = TargetRecord.model_validate(d)
            out[rec.record_id] = rec
        return out

    def load_comparisons(self, run_id: str) -> dict[str, ComparisonRecord]:
        out: dict[str, ComparisonRecord] = {}
        for d in self._iter(run_id, "comparisons.jsonl"):
            rec = ComparisonRecord.model_validate(d)
            out[rec.comparison_id] = rec
        return out

    def list_runs(self) -> list[RunManifest]:
        runs = []
        for d in self.root.iterdir() if self.root.exists() else []:
            if (d / "manifest.json").exists():
                try:
                    runs.append(self.load_manifest(d.name))
                except Exception:
                    continue
        return sorted(runs, key=lambda m: m.created_at, reverse=True)
