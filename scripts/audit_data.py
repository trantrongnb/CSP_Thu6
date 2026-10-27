"""Audit cached tensors and landmark coverage without changing class splits."""
import argparse
import json
from pathlib import Path

import numpy as np

from motion_part_fsl.datasets.metadata import read_metadata
from motion_part_fsl.datasets.landmark_cache import validate_cache
from motion_part_fsl.preprocessing.pipeline import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-source", action="store_true")
    args = parser.parse_args()
    report = {"splits": {}, "failures": [], "body_reference": {}, "source_verified": args.verify_source}
    hashes = {}
    config_hashes = set()
    for row in read_metadata(args.metadata):
        try:
            with np.load(args.cache_dir / f"{row['video_id']}.npz", allow_pickle=False) as archive:
                cache = {key: archive[key] for key in archive.files}
            validate_cache(cache, row["video_id"])
            config_hashes.add(str(cache["config_hash"].item()))
            source_hash = str(cache["source_sha256"].item())
            previous = hashes.setdefault(source_hash, row["video_id"])
            if previous != row["video_id"]:
                raise ValueError(f"Duplicate source bytes with {previous}")
            if args.verify_source:
                source = Path(row["path"])
                source = source if source.is_absolute() else args.metadata.parent / source
                if file_hash(source) != source_hash:
                    raise ValueError("Source changed since preprocessing")
            valid = cache["frame_valid"].astype(bool)
            entry = report["splits"].setdefault(row["split"], {"videos": 0, "all_hands_missing": 0,
                                                              "left_coverage_sum": 0., "right_coverage_sum": 0.})
            entry["videos"] += 1
            coverage = cache["hand_confidence"][valid].mean(0)
            entry["all_hands_missing"] += int(not coverage.any())
            entry["left_coverage_sum"] += float(coverage[0])
            entry["right_coverage_sum"] += float(coverage[1])
            reference = str(cache.get("body_reference", np.array("unknown")).item())
            report["body_reference"][reference] = report["body_reference"].get(reference, 0) + 1
        except (OSError, ValueError, KeyError) as exc:
            report["failures"].append({"video_id": row["video_id"], "reason": str(exc)})
    report["config_hashes"] = sorted(config_hashes)
    report["mixed_preprocessing"] = len(config_hashes) > 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Audit saved: {args.output}; failures={len(report['failures'])}")
    if report["failures"] or report["mixed_preprocessing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
