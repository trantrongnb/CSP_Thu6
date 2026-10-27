import argparse
from pathlib import Path

import cv2
import numpy as np

from motion_part_fsl.datasets.landmark_cache import validate_cache


def main():
    parser = argparse.ArgumentParser(description="Overlay anatomical L/R hands and torso reference on cached RGB")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    validate_cache(cache)
    height, width = cache["rgb"].shape[1:3]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 4., (width, height))
    if not writer.isOpened():
        raise OSError(f"Cannot write {args.output}")
    try:
        for i, rgb in enumerate(cache["rgb"]):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for r, (color, label) in enumerate((((0, 255, 0), "L"), ((0, 128, 255), "R"))):
                if cache["hand_confidence"][i, r] <= 0:
                    continue
                points = np.rint(cache["hands"][i, r] * [width, height]).astype(int)
                for point in points:
                    cv2.circle(bgr, tuple(point), 2, color, -1)
                cv2.putText(bgr, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
            origin = tuple(np.rint(cache["body_origin"][i] * [width, height]).astype(int))
            cv2.drawMarker(bgr, origin, (255, 255, 0), markerSize=8)
            cv2.putText(bgr, f"frame={cache['sampled_frame_ids'][i]} t={cache['timestamps'][i]:.3f} valid={int(cache['frame_valid'][i])}",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
            writer.write(bgr)
    finally:
        writer.release()


if __name__ == "__main__":
    main()
