import inspect

import pytest
import torch

from motion_part_fsl.models import MotionPartFSL
from motion_part_fsl.losses import episodic_loss
from conftest import tiny_model


@pytest.mark.parametrize("variant", [f"A{i}" for i in range(8)])
def test_all_variants_shapes_and_backward(episode_batch, variant):
    support, query, labels = episode_batch
    model = tiny_model(variant=variant)
    output = model(support, query)
    assert output["logits"].shape == (3, 3)
    assert torch.isfinite(output["logits"]).all()
    loss, _ = episodic_loss(output, labels, model.exclusive,
                           hard_pair_lambda=0.05 if variant == "A6" else 0.)
    loss.backward()
    assert torch.isfinite(loss)
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
    assert model.rgb.network[0].weight.grad.abs().sum() > 0
    if variant == "A5":
        assert model.motion.hands[0][0].weight.grad.abs().sum() > 0
        assert model.motion.relation[0].weight.grad.abs().sum() > 0
        assert model.pattern.motion_keys.weight.grad.abs().sum() > 0
        assert model.pattern.saliency.grad.abs().sum() > 0


def test_attention_normalizes_and_padding_is_masked(episode_batch):
    support, query, _ = episode_batch
    query = {key: value.clone() for key, value in query.items()}
    query["frame_valid"][:, -1] = False
    output = tiny_model().eval()(support, query)
    attention = output["query_attention"]
    torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention[..., 0]))
    assert (attention[..., -1] == 0).all()


def test_weights_use_support_only_and_queries_are_inductive(episode_batch):
    support, query, _ = episode_batch
    model = tiny_model().eval()
    original = model(support, query)
    changed_query = {k: v.clone() for k, v in query.items()}
    changed_query["rgb"][1:] = torch.randn_like(query["rgb"][1:]) * 10
    changed_query["quality"][1:] = 0
    changed = model(support, changed_query)
    torch.testing.assert_close(original["part_weights"], changed["part_weights"], rtol=0, atol=0)
    torch.testing.assert_close(original["logits"][0], changed["logits"][0], rtol=0, atol=0)
    assert "labels" not in inspect.signature(MotionPartFSL.forward).parameters


def test_episode_class_and_query_permutation_equivariance(episode_batch):
    support, query, _ = episode_batch
    model = tiny_model().eval()
    original = model(support, query)
    order = torch.tensor([2, 0, 1])
    permuted = model({k: v[order] for k, v in support.items()}, {k: v[order] for k, v in query.items()})
    torch.testing.assert_close(permuted["logits"], original["logits"][order][:, order], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(permuted["part_weights"], original["part_weights"][order], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("missing", ["left", "all"])
def test_missing_landmarks_finite_and_rgb_fallback(episode_batch, missing):
    support, query, _ = episode_batch
    support, query = ({k: v.clone() for k, v in batch.items()} for batch in (support, query))
    regions = [0, 2] if missing == "left" else [0, 1, 2]
    support["quality"][:, :, regions] = 0
    query["quality"][:, regions] = 0
    # Arbitrary invalid coordinates cannot leak through the motion encoder.
    support["hand_features"][:, :, 0] = 1000
    query["hand_features"][:, 0] = 1000
    model = tiny_model().eval()
    output = model(support, query)
    assert torch.isfinite(output["logits"]).all()
    assert (output["part_weights"][:, regions] == 0).all()
    if missing == "all":
        rgb = tiny_model(variant="A0").eval()
        rgb.load_state_dict(model.state_dict())
        torch.testing.assert_close(output["logits"], rgb(support, query)["logits"], atol=2e-6, rtol=2e-6)


def test_five_shot_and_mean_after_tokens(episode_batch):
    support, query, _ = episode_batch
    support = {k: v.repeat(1, 5, *([1] * (v.ndim - 2))) for k, v in support.items()}
    model = tiny_model(support_aggregation="tokens").eval()
    assert model(support, query)["logits"].shape == (3, 3)


def test_zero_motion_scale_is_independent_of_motion_inputs(episode_batch):
    support, query, _ = episode_batch
    model = tiny_model(lambda_motion=0).eval()
    original = model(support, query)
    changed = {key: value.clone() for key, value in query.items()}
    changed["hand_features"] = torch.randn_like(changed["hand_features"]) * 100
    changed["relation_features"] = torch.randn_like(changed["relation_features"]) * 100
    torch.testing.assert_close(original["logits"], model(support, changed)["logits"], atol=0, rtol=0)


def test_chunked_checkpointed_backbone_preserves_logits_and_gradients(episode_batch):
    support, query, labels = episode_batch
    reference = tiny_model()
    checkpointed = tiny_model(frame_batch_size=4, gradient_checkpointing=True)
    checkpointed.load_state_dict(reference.state_dict())
    expected, actual = reference(support, query), checkpointed(support, query)
    torch.testing.assert_close(actual["logits"], expected["logits"], atol=2e-6, rtol=2e-6)
    episodic_loss(actual, labels)[0].backward()
    episodic_loss(expected, labels)[0].backward()
    torch.testing.assert_close(checkpointed.rgb.network[0].weight.grad, reference.rgb.network[0].weight.grad,
                               atol=2e-6, rtol=2e-5)
