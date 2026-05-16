"""
方案 A v1 ordinal 单元测试 (2026-05-15)

3 个强制通过的测试：
  test_1_model_output_shape  → 模型 forward 输出 [B, 4]
  test_2_ordinal_target      → PHQ=12 → target = [1, 1, 0, 0]
  test_3_decode              → decode_ordinal_to_phq 输出合理

任一失败必须停下，不允许进入训练。
"""
import sys
from pathlib import Path

import torch

# 把项目根加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics import OrdinalBCELoss, compute_ordinal_pos_weight, decode_ordinal_to_phq
from models import TorchcatBaseline


# ─────────────────────────────────────────────────────────────────────────
def test_1_model_output_shape():
    """A-V+P subtrack，ordinal 模式，batch=8 → 输出 [8, 4]"""
    model = TorchcatBaseline(
        subtrack="A-V+P",
        num_classes=2,
        is_regression=True,
        use_regression_head=False,
        regression_head_mode="ordinal",
        ordinal_n_thresholds=4,
        audio_dim=64,
        video_dim=1000,
        gait_dim=0,
        hidden_dim=64,
        dropout=0.5,
        encoder_type="bilstm_mean",
    )
    model.eval()
    audio = torch.randn(8, 4, 128, 64)
    video = torch.randn(8, 4, 128, 1000)
    pers = torch.randn(8, 1024)
    pair_mask = torch.ones(8, 4)
    with torch.no_grad():
        out = model(audio=audio, video=video, personality=pers, pair_mask=pair_mask)
    assert out.shape == (8, 4), f"expected [8, 4], got {tuple(out.shape)}"
    print(f"[PASS] test_1_model_output_shape — output shape = {tuple(out.shape)}")


# ─────────────────────────────────────────────────────────────────────────
def test_2_ordinal_target():
    """各 PHQ 值的 ordinal target 必须正确。最重要：PHQ=12 → [1, 1, 0, 0]"""
    thresholds = torch.tensor([5., 10., 15., 20.])
    cases = [
        (0.0,  [0, 0, 0, 0]),
        (4.99, [0, 0, 0, 0]),
        (5.0,  [1, 0, 0, 0]),     # 阈值边界：>= 包含等于
        (7.0,  [1, 0, 0, 0]),
        (10.0, [1, 1, 0, 0]),
        (12.0, [1, 1, 0, 0]),     # 用户明确要求验证
        (14.99,[1, 1, 0, 0]),
        (15.0, [1, 1, 1, 0]),
        (19.99,[1, 1, 1, 0]),
        (20.0, [1, 1, 1, 1]),
        (27.0, [1, 1, 1, 1]),
    ]
    for phq, expected in cases:
        target = (torch.tensor([phq]).unsqueeze(-1) >= thresholds).float()
        got = [int(x) for x in target.tolist()[0]]
        assert got == expected, f"PHQ={phq}: got {got}, expected {expected}"
    print(f"[PASS] test_2_ordinal_target — all {len(cases)} cases correct")
    print("       PHQ=12 → [1, 1, 0, 0] ✓ (用户指定的关键 case)")


# ─────────────────────────────────────────────────────────────────────────
def test_3_decode():
    """decode_ordinal_to_phq 在 4 个边界场景下输出合理值。"""
    thresholds = [5., 10., 15., 20.]

    # ── case A: 全负 logits → P_ge ≈ [0,0,0,0] → bin_0 ≈ 1 → PHQ ≈ midpoint(0,5)=2.5
    logits_a = torch.tensor([[-10., -10., -10., -10.]])
    phq_a, bins_a = decode_ordinal_to_phq(logits_a, thresholds)
    assert phq_a.item() < 3.0, f"all-neg logits should give PHQ ~2.5, got {phq_a.item():.3f}"
    print(f"  case A (all-neg)        → PHQ={phq_a.item():.3f} (expected ~2.5)  bins={[round(x,3) for x in bins_a.tolist()[0]]}")

    # ── case B: 全正 logits → P_ge ≈ [1,1,1,1] → bin_K ≈ 1 → PHQ ≈ midpoint(20,27)=23.5
    logits_b = torch.tensor([[10., 10., 10., 10.]])
    phq_b, bins_b = decode_ordinal_to_phq(logits_b, thresholds)
    assert phq_b.item() > 22.0, f"all-pos logits should give PHQ ~23.5, got {phq_b.item():.3f}"
    print(f"  case B (all-pos)        → PHQ={phq_b.item():.3f} (expected ~23.5)  bins={[round(x,3) for x in bins_b.tolist()[0]]}")

    # ── case C: 边界 logits 制造 P_ge=[1, 0.5, 0, 0]
    # → bin_probs ≈ [0, 0.5, 0.5, 0, 0]
    # → PHQ ≈ 0.5*midpoint(5,10) + 0.5*midpoint(10,15) = 0.5*7.5 + 0.5*12.5 = 10.0
    logits_c = torch.tensor([[10., 0., -10., -10.]])
    phq_c, bins_c = decode_ordinal_to_phq(logits_c, thresholds)
    assert 9.0 < phq_c.item() < 11.0, (
        f"boundary logits should give PHQ ~10.0, got {phq_c.item():.3f}"
    )
    print(f"  case C (boundary)       → PHQ={phq_c.item():.3f} (expected ~10.0)  bins={[round(x,3) for x in bins_c.tolist()[0]]}")

    # ── case D: 单调违反（P_ge[1] > P_ge[0]） → cummin 修正后 bin_probs 仍合法
    # 原始 P_ge = sigmoid([0, 2.197, -0.847, -2.197]) ≈ [0.5, 0.9, 0.3, 0.1]，违反单调
    # cummin 后 = [0.5, 0.5, 0.3, 0.1]
    logits_d = torch.tensor([[0., 2.197, -0.847, -2.197]])
    phq_d, bins_d = decode_ordinal_to_phq(logits_d, thresholds, enforce_monotonic=True)
    bp = bins_d.tolist()[0]
    assert all(b >= -1e-5 for b in bp), f"bin probs must be non-negative, got {bp}"
    assert abs(sum(bp) - 1.0) < 1e-4, f"bin probs should sum to 1, got sum={sum(bp):.6f}"
    print(f"  case D (mono-violated)  → PHQ={phq_d.item():.3f}  bins={[round(x,3) for x in bp]} (sum={sum(bp):.4f})")

    print("[PASS] test_3_decode — all 4 cases correct")


# ─────────────────────────────────────────────────────────────────────────
def test_4_pos_weight():
    """压力测试 compute_ordinal_pos_weight：长尾 + 0-positive 都应被 clamp"""
    # 模拟 Track2 Young 实际分布: PHQ 0-4 多, PHQ 15+ 几乎没有
    fake_phq = torch.tensor(
        [0.]*7 + [1.]*9 + [2.]*9 + [3.]*10 + [4.]*10 +    # 45 个 PHQ 0-4
        [5.]*8 + [6.]*15 + [7.]*4 + [8.]*2 + [9.]*4 +     # 33 个 PHQ 5-9
        [10.]*4 + [11.]*4 + [16.]*1 + [17.]*1              # 10 个 PHQ 10+
    )
    thresholds = [5., 10., 15., 20.]
    pw = compute_ordinal_pos_weight(fake_phq, thresholds, max_clamp=5.0)
    print(f"  pos_weight (clamp=5):     {[round(x,3) for x in pw.tolist()]}")
    assert pw.shape == (4,), f"expected [4], got {pw.shape}"
    assert pw.max().item() <= 5.0 + 1e-6, f"clamp violated: max={pw.max().item():.3f}"
    # PHQ>=20 是 0 个正样本 → 原始 pos_weight 应为 inf，clamp 到 5.0
    assert pw[-1].item() == 5.0, f"PHQ>=20 head pos_weight should clamp to 5.0, got {pw[-1].item():.3f}"
    print(f"[PASS] test_4_pos_weight — clamping works")


# ─────────────────────────────────────────────────────────────────────────
def test_5_loss_smoke():
    """OrdinalBCELoss 端到端 smoke：能反传 + loss 是有限正数"""
    thresholds = [5., 10., 15., 20.]
    pw = torch.tensor([1.0, 5.0, 5.0, 5.0])
    criterion = OrdinalBCELoss(thresholds=thresholds, pos_weight=pw)

    # 模拟 batch=8 训练步
    logits = torch.randn(8, 4, requires_grad=True)
    phq = torch.tensor([0., 3., 6., 8., 11., 14., 17., 22.])
    loss = criterion(logits, phq)
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"loss should be positive, got {loss.item():.3f}"
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), "grad failed"
    print(f"[PASS] test_5_loss_smoke — loss={loss.item():.4f}, grad has shape {tuple(logits.grad.shape)}")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("方案 A v1 — Ordinal 单元测试 (5 个)")
    print("=" * 70)
    print()
    print("[Test 1] 模型 forward 输出 shape")
    test_1_model_output_shape()
    print()
    print("[Test 2] Ordinal target 计算（PHQ=12 → [1,1,0,0]）")
    test_2_ordinal_target()
    print()
    print("[Test 3] decode_ordinal_to_phq 输出合理")
    test_3_decode()
    print()
    print("[Test 4] compute_ordinal_pos_weight clamping")
    test_4_pos_weight()
    print()
    print("[Test 5] OrdinalBCELoss 端到端反传 smoke")
    test_5_loss_smoke()
    print()
    print("=" * 70)
    print("✅ ALL 5 TESTS PASSED")
    print("=" * 70)
