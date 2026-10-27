"""Numerical regression against the untouched local official TEAM checkout."""
import ast
from pathlib import Path
import types

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from motion_part_fsl.models.motion_guided_patterns import PatternAggregator
from motion_part_fsl.models.support_adaptation import adapt_support
from motion_part_fsl.models.matching import token_matching


ORIGINAL = Path(__file__).resolve().parents[1] / "references/team_original"


def original_matcher():
    # Compile the original methods directly, without importing timm/backbones.
    # This avoids changing the original checkout or reimplementing its formulas.
    tree = ast.parse((ORIGINAL / "model.py").read_text())
    methods = {}
    selected = {"CNN_FSHead": {"get_other"},
                "TEAM_pos": {"get_cum_dists_pos", "get_cum_dists_neg"},
                "TEAM": {"get_cum_dists"}}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in selected:
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and method.name in selected[cls.name]:
                    env = {"torch": torch, "F": F}
                    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ORIGINAL / "model.py"), "exec"), env)
                    methods[method.name] = env[method.name]
    matcher = type("OriginalMatcher", (nn.Module,), methods)()
    matcher.dropout = nn.Dropout(0.)
    return matcher


@pytest.mark.parametrize("shot", [1, 5])
def test_rgb_tokens_support_adaptation_and_logits_match_official_code(shot):
    torch.manual_seed(8)
    module = types.ModuleType("original_team_util")
    source = ORIGINAL / "model_util.py"
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    original = module.Discriminative_Pattern_Matching(4, d_model=32, n_head=4, d_k=8,
                                                     d_v=8, attn_dropout=0., dropout=0., seq_len=4).eval()
    ours = PatternAggregator(32, 4, 4, 4, dropout=0., attention_dropout=0.).eval()
    with torch.no_grad():
        ours.patterns.copy_(original.AP)
        ours.ln.load_state_dict(original.ln.state_dict())
        ours.mlp[0].load_state_dict(original.mlp.inp_fc.state_dict())
        ours.mlp[2].load_state_dict(original.mlp.out_fc.state_dict())
    # Original TEAM averages K frame sequences before DPM.
    support = torch.randn(3, shot, 4, 32).mean(1)
    query = torch.randn(2, 4, 32)
    dummy = lambda b: dict(motion=torch.zeros(b, 3, 4, 32), quality=torch.zeros(b, 3, 4),
                          frame_valid=torch.ones(b, 4, dtype=torch.bool), mode="rgb")
    s = ours(support, **dummy(3))
    q = ours(query, **dummy(2))
    op, on = original(support[None])
    op, on = F.relu(op), F.relu(on)
    qp, qn = original(query[:, None])
    qp, qn = F.relu(qp), F.relu(qn)
    torch.testing.assert_close(s["positive"][:, 0], op[0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(q["negative"][:, 0], qn[:, 0], atol=2e-6, rtol=2e-6)
    other = torch.stack([torch.cat([support[:c], support[c + 1:]]) for c in range(3)])[None]
    sim_p = torch.stack([F.cosine_similarity(op[0, c], torch.cat([op[0, :c], op[0, c + 1:]]), dim=-1)
                         for c in range(3)])[None]
    sim_n = torch.stack([F.cosine_similarity(on[0, c], torch.cat([on[0, :c], on[0, c + 1:]]), dim=-1)
                         for c in range(3)])[None]
    odp, odn = original(support[None], other, sim_p, sim_n)
    odp, odn = F.relu(odp).mean(2), F.relu(odn).mean(2)
    sp, sn = adapt_support(ours, s["context"], s["positive"], s["negative"], s["prior"])
    torch.testing.assert_close(sp[:, 0], odp[0], atol=3e-6, rtol=2e-6)
    torch.testing.assert_close(sn[:, 0], odn[0], atol=3e-6, rtol=2e-6)
    distances, positive_distance, negative_distance = original_matcher().get_cum_dists(odp, odn, op, on, qp, qn)
    pos, neg = token_matching(sp, sn, q["positive"], q["negative"], torch.ones(3, 1), token_reduction="sum")
    torch.testing.assert_close(pos, -positive_distance, atol=3e-6, rtol=2e-6)
    torch.testing.assert_close(neg, -negative_distance, atol=3e-6, rtol=2e-6)
    torch.testing.assert_close(pos + neg, -distances, atol=4e-6, rtol=2e-6)
