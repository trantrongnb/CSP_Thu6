import csv

import cv2
import numpy as np
import pytest

from motion_part_fsl.preprocessing.pipeline import main
from motion_part_fsl.datasets.landmark_cache import validate_cache


def test_decode_cache_padding_resume_and_stale_source_detection(tmp_path):
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10., (32, 24))
    assert writer.isOpened()
    for value in (30, 80, 130):
        writer.write(np.full((24, 32, 3), value, np.uint8))
    writer.release()
    metadata = tmp_path / "metadata.csv"
    row = dict(video_id="v1", path=str(video), gloss="hello", class_id="1", language="TEST", signer_id="", split="train")
    with metadata.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    args = ["--metadata", str(metadata), "--output", str(tmp_path / "cache"),
            "--frames", "4", "--image-size", "32", "--landmarks", "none"]
    main(args)
    with np.load(tmp_path / "cache/v1.npz", allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    validate_cache(cache)
    np.testing.assert_array_equal(cache["sampled_frame_ids"], [0, 1, 2, 2])
    np.testing.assert_array_equal(cache["frame_valid"], [1, 1, 1, 0])
    np.testing.assert_allclose(cache["timestamps"], [0., .1, .2, .2], atol=1e-6)
    assert (cache["hand_confidence"] == 0).all()
    before = (tmp_path / "cache/v1.npz").stat().st_mtime_ns
    main(args)
    assert (tmp_path / "cache/v1.npz").stat().st_mtime_ns == before
    with video.open("ab") as handle:
        handle.write(b"source_changed")
    with pytest.raises(ValueError, match="stale cache"):
        main(args)
