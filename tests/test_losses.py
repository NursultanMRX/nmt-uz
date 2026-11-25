import torch
import torch.nn.functional as F

from src.training.losses import stable_cross_entropy, stable_log_softmax


def test_matches_torch_on_ordinary_logits():
    torch.manual_seed(0)
    z = torch.randn(32, 100, dtype=torch.float64) * 3
    t = torch.randint(0, 100, (32,))
    assert torch.allclose(stable_log_softmax(z), F.log_softmax(z, -1), atol=1e-12)
    assert torch.allclose(
        stable_cross_entropy(z, t), F.cross_entropy(z, t, reduction="none"), atol=1e-12
    )


def test_label_smoothing_matches_torch():
    torch.manual_seed(0)
    z = torch.randn(16, 50, dtype=torch.float64)
    t = torch.randint(0, 50, (16,))
    ours = stable_cross_entropy(z, t, label_smoothing=0.1)
    ref = F.cross_entropy(z, t, reduction="none", label_smoothing=0.1)
    assert torch.allclose(ours, ref, atol=1e-12)


def test_no_softmax_collapse_in_fp32():
    # Margin 18 is past the fp32 cliff (~16.6): torch rounds the target gradient to 0.
    z = torch.tensor([[9.0, -9.0, -9.0]], requires_grad=True)
    t = torch.tensor([0])
    stable_cross_entropy(z, t).sum().backward()
    g = z.grad[0]
    assert g[0] < 0
    assert torch.allclose(g, torch.tensor([-3.046e-8, 1.523e-8, 1.523e-8]), rtol=1e-3)
    assert abs(g.sum()) < 1e-12

    z_ref = z.detach().clone().requires_grad_()
    F.cross_entropy(z_ref, t).backward()
    assert z_ref.grad[0, 0] == 0  # the bug this module exists to avoid


def test_stable_log_softmax_has_zero_sum_gradient_at_high_margin():
    z = torch.tensor([[9.0, -9.0, -9.0]], requires_grad=True)
    (-stable_log_softmax(z)[0, 0]).backward()
    assert z.grad[0, 0] < 0 and abs(z.grad.sum()) < 1e-12
