import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import cv2
import numpy as np

from ..datasets.metadata import read_metadata
from ..datasets.landmark_cache import validate_cache


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_frame_ids(count, frames):
    if count < 1 or frames < 2:
        raise ValueError("Require a nonempty video and T>=2")
    if count >= frames:
        return np.linspace(0, count - 1, frames).astype(np.int64)
    return np.r_[np.arange(count), np.full(frames - count, count - 1)].astype(np.int64)


def decode_video(path, frames=8, image_size=224, is_mirrored=False):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    decoded, times = [], []
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            decoded.append(bgr)
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        cap.release()
    ids = sample_frame_ids(len(decoded), frames)
    times = np.asarray(times, np.float64)
    if not np.isfinite(times).all() or (len(times) > 1 and (np.diff(times) <= 0).any()):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{path}: cannot establish real frame timestamps/FPS")
        times = np.arange(len(decoded), dtype=np.float64) / fps
    output, transforms = [], []
    for i in ids:
        rgb = cv2.cvtColor(decoded[i], cv2.COLOR_BGR2RGB)
        if is_mirrored:
            rgb = rgb[:, ::-1].copy()
        h, w = rgb.shape[:2]
        factor = min(image_size / w, image_size / h)
        nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
        left, top = (image_size - nw) // 2, (image_size - nh) // 2
        canvas = np.zeros((image_size, image_size, 3), np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh))
        output.append(canvas)
        transforms.append([nw / w, nh / h, left, top])
    return {"rgb": np.stack(output), "timestamps": times[ids], "sampled_frame_ids": ids,
            "frame_valid": np.r_[True, np.diff(ids) > 0], "fps": np.array(fps),
            "letterbox_transform": np.asarray(transforms, np.float32),
            "is_mirrored": np.array(is_mirrored)}


def extract_landmarks(rgb, frame_valid, backend="holistic"):
    t = len(rgb)
    hands, confidence = np.zeros((t, 2, 21, 2), np.float32), np.zeros((t, 2), np.float32)
    centers, scales = [], []
    if backend == "holistic":
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError("Holistic requires mediapipe==0.10.21; use an isolated environment with motion-part-fsl[landmarks]")
        with mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=0.5,
                                             min_tracking_confidence=0.5) as detector:
            for i, frame in enumerate(rgb):
                if not frame_valid[i]:
                    continue
                result = detector.process(frame)
                for r, points in enumerate((result.left_hand_landmarks, result.right_hand_landmarks)):
                    if points is not None:
                        xy = np.array([(p.x, p.y) for p in points.landmark], np.float32)
                        if np.isfinite(xy).all():
                            hands[i, r], confidence[i, r] = xy, 1.0  # detected, no invented confidence
                if result.pose_landmarks is not None:
                    shoulders = [result.pose_landmarks.landmark[j] for j in (11, 12)]
                    xy = np.array([(p.x, p.y) for p in shoulders], np.float32)
                    width = np.linalg.norm(xy[0] - xy[1])
                    if np.isfinite(xy).all() and min(p.visibility for p in shoulders) >= 0.5 and width > 1e-3:
                        centers.append(xy.mean(0))
                        scales.append(width)
    elif backend != "none":
        raise ValueError(f"Unknown landmark backend {backend}")
    # Stable per-video torso coordinates prevent missing shoulders from causing
    # artificial jumps in velocity. Image coordinates are an explicit fallback.
    origin = np.median(centers, axis=0).astype(np.float32) if centers else np.array([0.5, 0.5], np.float32)
    scale = np.float32(np.median(scales)) if scales else np.float32(1.0)
    return {"hands": hands, "hand_confidence": confidence,
            "body_origin": np.broadcast_to(origin, (t, 2)).copy(), "body_scale": np.full(t, scale),
            "body_reference": np.array("video_median_shoulders" if centers else "image_center_unit_scale")}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cache synchronized RGB/2D hands for Motion–Part FSL")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--landmarks", choices=["holistic", "none"], default="holistic")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.frames < 2 or args.image_size < 16 or (args.limit is not None and args.limit < 1):
        raise ValueError("Invalid frame count, image size or limit")
    rows = read_metadata(args.metadata)
    config = {"schema": 1, "frames": args.frames, "image_size": args.image_size,
              "landmarks": args.landmarks, "coordinates": "2D_stable_video_torso_or_image",
              "mediapipe_version": version("mediapipe") if args.landmarks == "holistic" else None,
              "opencv_version": cv2.__version__, "sampling": "uniform_pad_end_mask"}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "preprocessing_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["config_hash"] != config_hash:
        raise ValueError("Changed preprocessing; use a new cache directory")
    manifest.write_text(json.dumps({**config, "config_hash": config_hash}, indent=2))
    for index, row in enumerate(rows[:args.limit]):
        path = Path(row["path"])
        path = path if path.is_absolute() else args.metadata.parent / path
        source_hash = file_hash(path)
        mirror = row.get("is_mirrored", "false").lower()
        if mirror not in {"true", "false", "0", "1", ""}:
            raise ValueError(f"Invalid is_mirrored for {row['video_id']}")
        mirrored = mirror in {"true", "1"}
        target = args.output / f"{row['video_id']}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as archive:
                cache = {key: archive[key] for key in archive.files}
            validate_cache(cache, row["video_id"])
            if (str(cache["config_hash"].item()) != config_hash
                or str(cache["source_sha256"].item()) != source_hash or bool(cache["is_mirrored"]) != mirrored):
                raise ValueError(f"{target}: stale cache; use a new output directory")
            continue
        start = time.perf_counter()
        data = decode_video(path, args.frames, args.image_size, mirrored)
        data.update(extract_landmarks(data["rgb"], data["frame_valid"], args.landmarks))
        data.update(config_hash=np.array(config_hash), source_sha256=np.array(source_hash),
                    preprocessing_seconds=np.array(time.perf_counter() - start))
        validate_cache(data, row["video_id"])
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **data)
        temporary.replace(target)
        print(f"{index + 1}: {row['video_id']} cached; hand coverage={data['hand_confidence'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
