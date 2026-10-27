import torch

from motion_part_fsl.models.part_weighting import SupportPartWeights


def test_nearest_competitor_and_uniform_shrinkage():
    x = torch.tensor([[[[1., 0.]], [[1., 0.]]],
                      [[[0., 1.]], [[1., 0.]]],
                      [[[-1., 0.]], [[1., 0.]]]])
    module = SupportPartWeights(2, reliability_lambda=0, uniform_shrinkage=0)
    weights, separation = module(x, torch.ones(3, 2))
    torch.testing.assert_close(separation[:, 0], torch.ones(3))
    assert (separation[:, 1] == 0).all()
    assert (weights[:, 0] > weights[:, 1]).all()
    uniform, _ = SupportPartWeights(2, uniform_shrinkage=1)(x, torch.ones(3, 2))
    torch.testing.assert_close(uniform, torch.full((3, 2), .5))


def test_all_invalid_and_no_valid_competitor_have_zero_weight():
    x = torch.randn(3, 3, 4, 8, requires_grad=True)
    reliability = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    weights, separation = SupportPartWeights()(x, reliability)
    assert (weights == 0).all() and torch.isfinite(separation).all()
    weights.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_episode_composition_changes_weights_without_changing_anchor():
    torch.manual_seed(42)
    x = torch.randn(3, 3, 4, 8)
    module = SupportPartWeights(reliability_lambda=0)
    before, _ = module(x, torch.ones(3, 3))
    x[1, 0] = x[0, 0]
    after, _ = module(x, torch.ones(3, 3))
    assert after[0, 0] < before[0, 0]
