"""Synthetic sanity experiment through the production train/evaluate code path."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .datasets.video_dataset import VideoDataset, load_episode
from .datasets.episodic_sampler import EpisodeSampler
from .engine import build_model, train_main, evaluate_main
from .losses import episodic_loss
from .utils.seed import seed_everything


def synthetic_cache(label=0, sample=0, frames=4, image_size=24):
    rng = np.random.default_rng(100 * label + sample)
    rgb = np.full((frames, image_size, image_size, 3), 30, np.uint8)
    rgb[..., label % 3] = 180
    rgb = np.clip(rgb.astype(np.int16) + rng.integers(-4, 5, rgb.shape), 0, 255).astype(np.uint8)
    hands = np.zeros((frames, 2, 21, 2), np.float32)
    shape = np.linspace(-0.05, 0.05, 21)
    for t in range(frames):
        hands[t, 0, :, 0] = 0.3 + shape + t * (label + 1) * 0.01
        hands[t, 0, :, 1] = 0.4 + shape
        hands[t, 1, :, 0] = 0.65 + shape
        hands[t, 1, :, 1] = 0.45 + shape + t * (3 - label % 3) * 0.01
    signature = hashlib.sha256(f"synthetic:{label}:{sample}".encode()).hexdigest()
    return {"rgb": rgb, "hands": hands, "hand_confidence": np.ones((frames, 2), np.float32),
            "body_origin": np.tile(np.array([0.5, 0.5], np.float32), (frames, 1)),
            "body_scale": np.full(frames, 0.4, np.float32), "timestamps": np.arange(frames) / 10.,
            "sampled_frame_ids": np.arange(frames), "frame_valid": np.ones(frames, bool),
            "config_hash": np.array(f"synthetic_v1_t{frames}_h{image_size}"), "source_sha256": np.array(signature)}


def make_synthetic_dataset(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    cache_dir = root / "cache"
    cache_dir.mkdir()
    rows = []
    for split_index, split in enumerate(("train", "val", "test")):
        for c in range(3):
            for sample in range(4):
                vid = f"{split}_{c}_{sample}"
                rows.append(dict(video_id=vid, path=f"{vid}.mp4", gloss=f"sign_{split}_{c}",
                                 class_id=str(split_index * 3 + c), language="SYNTHETIC", signer_id="", split=split))
                np.savez_compressed(cache_dir / f"{vid}.npz", **synthetic_cache(c, sample))
    metadata = root / "metadata.csv"
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metadata, cache_dir


def small_config(metadata, cache_dir, output):
    return {"seed": 42, "output_dir": str(output),
            "data": {"metadata": str(metadata), "cache_dir": str(cache_dir)},
            "model": {"backbone": "tiny", "pretrained": False, "feature_dim": 32,
                      "pattern_tokens": 4, "frames": 4, "heads": 4, "dropout": 0.,
                      "attention_dropout": 0., "variant": "A5"},
            "training": {"n_way": 3, "k_shot": 1, "queries_per_class": 1, "max_epochs": 2,
                         "episodes_per_epoch": 3, "lr": 0.003, "weight_decay": 0.,
                         "grad_clip": 1., "amp": False, "log_every": 3},
            "evaluation": {"val_episodes_file": str(output.parent / "val_episodes.json"),
                           "val_episodes": 3, "seed": 123}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthetic Motion–Part FSL overfit and CLI check")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a new smoke output directory")
    if args.steps < 1:
        raise ValueError("Steps must be positive")
    torch.set_num_threads(2)
    seed_everything(42)
    metadata, cache_dir = make_synthetic_dataset(args.output / "data")
    config = small_config(metadata, cache_dir, args.output / "training")
    dataset = VideoDataset(metadata, cache_dir, "train")
    episode = EpisodeSampler(dataset.class_to_videos, 3, 1, 1).sample()
    support, query, labels = load_episode(dataset, episode, torch.device("cpu"))
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(support, query)
        loss, _ = episodic_loss(output, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite synthetic loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        accuracy = (model(support, query)["logits"].argmax(-1) == labels).float().mean().item()
    path = args.output / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    train_main(["--config", str(path), "--device", "cpu"])
    test = VideoDataset(metadata, cache_dir, "test")
    episodes = args.output / "test_episodes.json"
    EpisodeSampler(test.class_to_videos, 3, 1, 1, 123).save(episodes, 3, test)
    evaluate_main(["--checkpoint", str(args.output / "training/best.pt"), "--episodes", str(episodes),
                   "--output", str(args.output / "test.json"), "--device", "cpu"])
    summary = {"data": "synthetic_only", "initial_loss": losses[0], "final_loss": losses[-1],
               "overfit_accuracy": accuracy, "steps": args.steps}
    (args.output / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
