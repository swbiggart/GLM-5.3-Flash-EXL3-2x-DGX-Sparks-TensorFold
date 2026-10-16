"""Zig patch 0008: five dense attention chunks keep the Python reference's bits at TP=2.

These are real Triton launches on synthetic FP8 caches, not a native AOT or served-model qualification.
The native host test separately checks launch matching. Run through scripts/test-gpu.sh in the Python image.
"""

import pytest
import torch

from tensorfold.families.glm5_next.cuda import kv8, latent


@pytest.mark.parametrize("hb", [16, 32])
@pytest.mark.parametrize("pos,rows,old_nch", [
    (0, 1, 1), (0, 15, 1), (0, 16, 1), (0, 63, 1), (0, 64, 1), (0, 512, 1),
    (512, 1, 2), (512, 512, 2),
    (1024, 1, 3), (1024, 16, 3), (1024, 512, 3), (1152, 41, 3),
    (1536, 1, 4), (1536, 512, 4),
    (2048, 1, 5), (2048, 2, 5), (2048, 3, 5),
    (2048, 3, 6), (2048, 3, 7), (2048, 3, 8), (2048, 3, 9), (2048, 3, 37),
])
def test_five_chunks_preserve_dense_attention_bits(pos, rows, old_nch, hb):
    """Poison partials first: every extra chunk must be written, then act as a neutral merge input."""
    g = torch.Generator().manual_seed(1000 + pos + rows)
    h, lw = 32, 512
    values = (torch.randn(pos + rows, lw, generator=g) * 2).to(torch.bfloat16).cuda()
    cache = kv8.quantize_rows(values)
    qa = (torch.randn(rows, h, lw, generator=g) * 0.05).to(torch.bfloat16).cuda()
    origin = torch.tensor([pos], dtype=torch.int32, device="cuda")
    scratch = latent.LatentScratch(rows, h, max(5, old_nch), "cuda")
    outputs = []
    for nch in (old_nch, 5):
        for partial in (scratch.po, scratch.pm, scratch.pl):
            partial.fill_(float("nan"))
        out = torch.full_like(qa, float("nan"))
        latent.attention(qa, cache, origin, scratch, scale=0.0625, nch=nch, out=out, hb=hb)
        assert torch.isfinite(out).all()
        empty = torch.arange(nch, device="cuda")[:, None] * 512 > (
            pos + torch.arange(rows, device="cuda")[None, :])
        po = scratch.po[:nch * rows * h * lw].view(nch, rows, h, lw)
        pm = scratch.pm[:nch * rows * h].view(nch, rows, h)
        pl = scratch.pl[:nch * rows * h].view(nch, rows, h)
        assert torch.equal(po[empty], torch.zeros_like(po[empty]))
        assert torch.equal(pl[empty], torch.zeros_like(pl[empty]))
        assert torch.isneginf(pm[empty]).all()
        outputs.append(out)
    assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
