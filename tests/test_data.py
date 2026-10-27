import json

import numpy as np
import pytest

from motion_part_fsl.datasets.landmark_cache import kinematic_features, validate_cache
from motion_part_fsl.datasets.video_dataset import VideoDataset
from motion_part_fsl.datasets.episodic_sampler import EpisodeSampler, load_episodes
from motion_part_fsl.preprocessing.pipeline import sample_frame_ids
from motion_part_fsl.smoke import synthetic_cache


def test_velocity_uses_real_dt_and_missing_never_creates_motion():
    cache = synthetic_cache()
    cache["body_scale"][:] = 1
    cache["hands"][..., 0] = np.array([0., .1, .3, .6])[:, None, None]
    cache["timestamps"] = np.array([0., .1, .3, .6])
    features = kinematic_features(cache)
    np.testing.assert_allclose(features["hand_features"][0, 1:, 42:84:2], 1., atol=1e-6)
    cache["hand_confidence"][1, 0] = 0
    features = kinematic_features(cache)
    assert (features["hand_features"][0, 1:3, 42:84] == 0).all()
    assert (features["quality"][2, 1:2] == 0).all()


def test_padding_and_cache_validation():
    np.testing.assert_array_equal(sample_frame_ids(2, 4), [0, 1, 1, 1])
    cache = synthetic_cache()
    cache["frame_valid"][1] = False
    with pytest.raises(ValueError, match="padding mask"):
        validate_cache(cache)


def test_fixed_episodes_no_leakage_and_tamper_detection(synthetic_data, tmp_path):
    dataset = VideoDataset(*synthetic_data, "val")
    first = EpisodeSampler(dataset.class_to_videos, 3, 1, 1, seed=11)
    second = EpisodeSampler(dataset.class_to_videos, 3, 1, 1, seed=11)
    assert first.sample() == second.sample()
    path = tmp_path / "episodes.json"
    first.save(path, 3, dataset)
    document = load_episodes(path, dataset)
    for episode in document["episodes"]:
        assert not set(episode["support"]) & set(episode["query"])
    document["episodes"][0]["query"][0] = document["episodes"][0]["support"][0]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="leakage"):
        load_episodes(path, dataset)


def test_metadata_class_leakage_is_rejected(synthetic_data):
    metadata, cache_dir = synthetic_data
    content = metadata.read_text().replace("sign_val_0,3,", "sign_train_0,0,")
    metadata.write_text(content)
    with pytest.raises(ValueError, match="leakage"):
        VideoDataset(metadata, cache_dir, "train")
