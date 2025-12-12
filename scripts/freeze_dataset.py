"""
Freeze the CURRENT data/processed/ datasets into a new immutable snapshot.

This is for when you manually tweak datasets (e.g., add Washington prompts)
and you want to preserve the exact version used for experiments.

Output:
  data/processed_versions/local_<timestamp>/
    - forget.json
    - retain.json
    - probe_train.json
    - manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


DATASET_FILES = ("forget.json", "retain.json", "probe_train.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_dir = repo_root / "data" / "processed"
    versions_root = repo_root / "data" / "processed_versions"
    versions_root.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Freeze current data/processed/ into a versioned snapshot.")
    parser.add_argument(
        "--name",
        default=None,
        help="Optional custom snapshot folder name (default: local_<timestamp>).",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional note to store in the manifest (e.g., 'Added Washington prompts').",
    )
    args = parser.parse_args()

    missing = [fn for fn in DATASET_FILES if not (processed_dir / fn).exists()]
    if missing:
        raise SystemExit(f"Missing dataset files in {processed_dir}: {missing}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = args.name or f"local_{ts}"
    out_dir = versions_root / version_name
    if out_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing snapshot dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "dataset_version": version_name,
        "created_at": datetime.now().isoformat(),
        "source_dir": str(processed_dir.relative_to(repo_root)),
        "note": args.note,
        "files": {},
    }

    for fn in DATASET_FILES:
        src = processed_dir / fn
        dst = out_dir / fn
        shutil.copy2(src, dst)
        manifest["files"][fn] = {
            "sha256": sha256_file(dst),
            "num_records": len(json.loads(dst.read_text(encoding="utf-8"))),
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen dataset snapshot written to: {out_dir.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

