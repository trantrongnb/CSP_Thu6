"""Metadata validation shared in design with the historical MFPA baseline."""
import csv
from pathlib import Path


REQUIRED_COLUMNS = {"video_id", "path", "gloss", "class_id", "language", "signer_id", "split"}


def read_metadata(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS <= set(reader.fieldnames or []):
            raise ValueError(f"Metadata requires columns {sorted(REQUIRED_COLUMNS)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Metadata is empty")
    ids, paths, class_splits, gloss_splits = set(), set(), {}, {}
    for row in rows:
        vid = row["video_id"]
        if not vid or vid in ids or Path(vid).name != vid or vid in {".", ".."}:
            raise ValueError(f"Duplicate/unsafe video_id: {vid}")
        ids.add(vid)
        video_path = Path(row["path"])
        resolved = str((video_path if video_path.is_absolute() else path.parent / video_path).resolve())
        if not row["path"] or resolved in paths:
            raise ValueError(f"Repeated or empty source video path: {row['path']}")
        paths.add(resolved)
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {row['split']}")
        if not row["class_id"] or not row["gloss"].strip() or not row["language"].strip():
            raise ValueError(f"Empty class/gloss/language for {vid}")
        row["class_key"] = f"{row['language']}:{row['class_id']}"
        for table, key in ((class_splits, row["class_key"]),
                           (gloss_splits, (row["language"], row["gloss"].strip().casefold()))):
            previous = table.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError(f"Class/gloss leakage across splits: {key}")
    return rows

