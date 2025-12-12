"""
Activate a frozen dataset snapshot by copying it into data/processed/.

This repo stores immutable snapshots under:
  data/processed_versions/<version>/

This script copies:
  - forget.json
  - retain.json
  - probe_train.json

into:
  data/processed/

Safety:
  - Refuses to overwrite existing data/processed/*.json unless --overwrite is set.
  - Optional --backup will snapshot the current data/processed/ first.
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


def snapshot_processed(src_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "source_dir": str(src_dir),
        "files": {},
    }
    for fn in DATASET_FILES:
        src = src_dir / fn
        if not src.exists():
            continue
        dst = dest_dir / fn
        shutil.copy2(src, dst)
        manifest["files"][fn] = {
            "sha256": sha256_file(dst),
            "num_records": len(json.loads(dst.read_text(encoding="utf-8"))),
        }
    (dest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    versions_root = repo_root / "data" / "processed_versions"
    processed_dir = repo_root / "data" / "processed"

    parser = argparse.ArgumentParser(description="Activate a frozen dataset snapshot.")
    parser.add_argument(
        "version",
        help="Dataset version folder under data/processed_versions/ (e.g. v3_ca81135).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data/processed/*.json if they differ.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Before overwriting, snapshot current data/processed/ into data/processed_versions/backup_<timestamp>/",
    )

    args = parser.parse_args()

    src_dir = versions_root / args.version
    if not src_dir.exists():
        raise SystemExit(f"Dataset version not found: {src_dir}")

    missing = [fn for fn in DATASET_FILES if not (src_dir / fn).exists()]
    if missing:
        raise SystemExit(f"Dataset version is missing files: {missing}\nIn: {src_dir}")

    processed_dir.mkdir(parents=True, exist_ok=True)

    # Decide whether we're allowed to overwrite
    existing = [processed_dir / fn for fn in DATASET_FILES if (processed_dir / fn).exists()]
    if existing and not args.overwrite:
        # If contents already match, do nothing.
        all_match = True
        for fn in DATASET_FILES:
            dst = processed_dir / fn
            src = src_dir / fn
            if dst.exists() and sha256_file(dst) == sha256_file(src):
                continue
            all_match = False
            break
        if all_match:
            print(f"data/processed/ already matches version '{args.version}'. Nothing to do.")
            return

        raise SystemExit(
            "Refusing to overwrite existing data/processed/*.json.\n"
            "Re-run with --overwrite (and optionally --backup) if you really want to switch."
        )

    if existing and args.overwrite and args.backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = versions_root / f"backup_{ts}"
        snapshot_processed(processed_dir, backup_dir)
        print(f"Backed up current data/processed/ to: {backup_dir.relative_to(repo_root)}")

    # Copy files
    for fn in DATASET_FILES:
        shutil.copy2(src_dir / fn, processed_dir / fn)

    print(f"Activated dataset version '{args.version}' → data/processed/")


if __name__ == "__main__":
    main()

