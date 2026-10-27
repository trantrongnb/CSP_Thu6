import pytest
import torch

from motion_part_fsl.datasets.video_dataset import VideoDataset, load_episode
from motion_part_fsl.datasets.episodic_sampler import EpisodeSampler
from motion_part_fsl.smoke import make_synthetic_dataset


torch.set_num_threads(2)


@pytest.fixture
def synthetic_data(tmp_path):
    return make_synthetic_dataset(tmp_path / "data")


@pytest.fixture
def episode_batch(synthetic_data):
    dataset = VideoDataset(*synthetic_data, "train")
    episode = EpisodeSampler(dataset.class_to_videos, 3, 1, 1).sample()
    return load_episode(dataset, episode, torch.device("cpu"))


def tiny_model(**overrides):
    from motion_part_fsl.models import MotionPartFSL
    options = dict(backbone="tiny", pretrained=False, feature_dim=32, pattern_tokens=4,
                   frames=4, heads=4, dropout=0., attention_dropout=0.)
    return MotionPartFSL(**{**options, **overrides})
