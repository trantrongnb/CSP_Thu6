import numpy as np

from ..models.motion_encoder import HAND_DIM, RELATION_DIM


def _derivative(values, valid, timestamps):
    out = np.zeros_like(values)
    ok = np.zeros_like(valid, dtype=bool)
    dt = np.diff(timestamps)
    pair_ok = valid[1:] & valid[:-1]
    pair_ok &= (dt > 0).reshape((-1,) + (1,) * (valid.ndim - 1))
    denom = np.maximum(dt, 1e-8).reshape((-1,) + (1,) * (values.ndim - 1))
    out[1:] = np.where(pair_ok[..., None, None] if values.ndim == valid.ndim + 2 else pair_ok[..., None],
                       np.diff(values, axis=0) / denom, 0)
    ok[1:] = pair_ok
    return out, ok


def kinematic_features(cache):
    """Timestamp-aware 2D kinematics. Missing frames cannot create velocity."""
    hands = cache["hands"].astype(np.float32)
    confidence = cache["hand_confidence"].astype(np.float32)
    frame_valid = cache["frame_valid"].astype(bool)
    valid = (confidence > 0) & frame_valid[:, None]
    origin, scale = cache["body_origin"], cache["body_scale"]
    # The reference is stable per video, estimated only from that video's pose.
    normalized = (hands - origin[:, None, None]) / scale[:, None, None, None]
    normalized = np.where(valid[..., None, None], normalized, 0).astype(np.float32)
    velocity, vv = _derivative(normalized, valid, cache["timestamps"])
    acceleration, av = _derivative(velocity, vv, cache["timestamps"])
    relative = normalized - normalized[:, :, :1]
    masks = np.stack([valid, vv, av], axis=-1).astype(np.float32)
    hand_features = np.concatenate([x.reshape(len(hands), 2, 42)
                                    for x in (normalized, velocity, acceleration, relative)] + [masks], axis=-1)
    hand_features = hand_features.transpose(1, 0, 2)
    center = normalized.mean(2)
    relation_valid = valid.all(1)
    delta = center[:, 0] - center[:, 1]
    delta = np.where(relation_valid[:, None], delta, 0)
    delta_velocity, rv = _derivative(delta, relation_valid, cache["timestamps"])
    direction = normalized[:, :, 9] - normalized[:, :, 0]
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = direction / np.maximum(norm, 1e-6)
    # Relative palm direction encoded as dot/cross; no inferred 3D contact.
    orientation = np.stack([(direction[:, 0] * direction[:, 1]).sum(-1),
                            direction[:, 0, 0] * direction[:, 1, 1] - direction[:, 0, 1] * direction[:, 1, 0]], -1)
    relation = np.concatenate([delta, np.linalg.norm(delta, axis=-1, keepdims=True), delta_velocity,
                               orientation, np.stack([relation_valid, rv, relation_valid], -1)], -1)
    relation = np.where(relation_valid[:, None], relation, 0).astype(np.float32)
    confidence = confidence * valid
    quality = np.concatenate([confidence, confidence.min(1, keepdims=True)], -1).T
    assert hand_features.shape[-1] == HAND_DIM and relation.shape[-1] == RELATION_DIM
    return {"hand_features": hand_features, "relation_features": relation, "quality": quality,
            "frame_valid": frame_valid}


def validate_cache(cache, video_id="cache"):
    keys = {"rgb", "hands", "hand_confidence", "body_origin", "body_scale", "timestamps",
            "frame_valid", "sampled_frame_ids", "config_hash", "source_sha256"}
    if not keys <= cache.keys():
        raise ValueError(f"{video_id}: missing cache keys {sorted(keys - cache.keys())}")
    t = len(cache["timestamps"])
    expected = {"hands": (t, 2, 21, 2), "hand_confidence": (t, 2), "body_origin": (t, 2),
                "body_scale": (t,), "timestamps": (t,), "frame_valid": (t,), "sampled_frame_ids": (t,)}
    rgb = cache["rgb"]
    if t < 2 or rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] != t or rgb.shape[-1] != 3 or rgb.shape[1] != rgb.shape[2]:
        raise ValueError(f"{video_id}: invalid uint8 RGB [T,H,H,3]")
    for key, shape in expected.items():
        if cache[key].shape != shape or not np.isfinite(cache[key]).all():
            raise ValueError(f"{video_id}: invalid {key}")
    if not np.isfinite(cache["timestamps"]).all() or (cache["body_scale"] <= 0).any():
        raise ValueError(f"{video_id}: invalid timestamps/body scale")
    if not np.isin(cache["frame_valid"], [0, 1]).all() or not cache["frame_valid"].any():
        raise ValueError(f"{video_id}: invalid frame mask")
    ids = cache["sampled_frame_ids"]
    if not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any() or (np.diff(ids) < 0).any():
        raise ValueError(f"{video_id}: nonchronological frame IDs")
    expected_valid = np.r_[True, np.diff(ids) > 0]
    if not np.array_equal(cache["frame_valid"].astype(bool), expected_valid):
        raise ValueError(f"{video_id}: padding mask disagrees with sampled frame IDs")
    timestamps = cache["timestamps"]
    if (np.diff(timestamps) < 0).any() or (np.diff(timestamps)[np.diff(ids) > 0] <= 0).any():
        raise ValueError(f"{video_id}: timestamps must increase on genuine frames")
    q = cache["hand_confidence"]
    if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any() or (q[~expected_valid] != 0).any():
        raise ValueError(f"{video_id}: invalid landmark confidence")
