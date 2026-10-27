import hashlib
from pathlib import Path

import numpy as np
import torch

from .landmark_cache import kinematic_features, validate_cache
from .metadata import read_metadata


class VideoDataset:
    """One within-dataset experiment, using the same RGB/landmark cache for A0–A7."""

    def __init__(self, metadata, cache_dir, split):
        self.metadata, self.cache_dir, self.split = Path(metadata), Path(cache_dir), split
        all_rows = read_metadata(self.metadata)
        if len({r["language"] for r in all_rows}) != 1:
            raise ValueError("Use separate WLASL/VSL metadata for within-dataset experiments")
        self.rows = [r for r in all_rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No samples in {split}")
        self.by_id = {r["video_id"]: r for r in self.rows}
        self.class_to_videos, self.config_hash = {}, None
        signatures = []
        for row in self.rows:
            vid = row["video_id"]
            with np.load(self.cache_dir / f"{vid}.npz", allow_pickle=False) as cache:
                signature = str(cache["config_hash"].item())
                if self.config_hash is None:
                    self.config_hash = signature
                if signature != self.config_hash:
                    raise ValueError(f"{vid}: mixed preprocessing configurations")
                signatures.append(vid + ":" + str(cache["source_sha256"].item()))
            self.class_to_videos.setdefault(row["class_key"], []).append(vid)
        digest = self.metadata.read_bytes() + self.config_hash.encode() + "\n".join(signatures).encode()
        self.fingerprint = hashlib.sha256(digest).hexdigest()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        vid = self.rows[index]["video_id"] if isinstance(index, int) else index
        if vid not in self.by_id:
            raise KeyError(f"{vid}: video outside {self.split}")
        with np.load(self.cache_dir / f"{vid}.npz", allow_pickle=False) as archive:
            cache = {key: archive[key] for key in archive.files}
        validate_cache(cache, vid)
        if str(cache["config_hash"].item()) != self.config_hash:
            raise ValueError(f"{vid}: cache changed after dataset construction")
        features = kinematic_features(cache)
        rgb = cache["rgb"].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        # Fixed ImageNet normalization, shared across datasets/variants.
        mean = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
        std = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
        features["rgb"] = (rgb - mean) / std
        return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in features.items()}


def load_episode(dataset, episode, device):
    support_ids, query_ids = episode["support"], episode["query"]
    if len(set(support_ids + query_ids)) != len(support_ids + query_ids):
        raise ValueError("Repeated videos or support/query leakage")
    n = len(episode["classes"])
    if len(support_ids) % n:
        raise ValueError("Support must be class-major N*K")
    k = len(support_ids) // n
    samples = [dataset[vid] for vid in support_ids + query_ids]
    batch = {key: torch.stack([s[key] for s in samples]).to(device) for key in samples[0]}
    support = {key: value[:n * k].reshape(n, k, *value.shape[1:]) for key, value in batch.items()}
    query = {key: value[n * k:] for key, value in batch.items()}
    labels = torch.tensor(episode["query_labels"], dtype=torch.long, device=device)
    return support, query, labels
