from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    cov = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    denom = y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2
    return safe_float(2 * cov / denom) if denom > 1e-10 else 0.0


# =============================================================================
#  修复 #1 (Bug 根本层): CCCLoss + 方差惩罚项
# =============================================================================
class CCCLoss(nn.Module):
    """
    可微分 CCC 损失 + 方差惩罚项 + MSE 联合损失。

    【档位 A 修复】2026-05-13
    旧版只有 CCC + var_penalty 时模型仍然塌缩到 [2,8] 窄带：
      - CCC 对绝对尺度不敏感（只看相关性），允许"猜均值"低损失
      - var_penalty 阈值 min_var=0.05 对应 std≥0.22(log1p)，模型刚翻过去就停
      - 没有任何项强制"预测高分样本要给高分"，重症被系统性低估

    【本次改动】
      1. mse_weight=0.5 加入 MSE 项：强制绝对值对齐，重罚高分被预测低
      2. min_var 0.05 → 0.20：阈值对应 std≥0.45(log1p)，逼模型展开方差
      3. var_weight 0.5 → 1.0：方差惩罚不再被 CCC 项盖过

    【参数说明】
    var_weight : 方差惩罚系数，默认 1.0
    min_var    : 预测方差的最低期望值，默认 0.20
                 (log1p 空间下接近真实方差的 30~40%，逼分布展开)
    mse_weight : MSE 联合损失系数，默认 0.5
                 (CCC loss 量级 ~1.0，log1p 空间 MSE 量级 ~0.3，0.5 平衡)
    """

    def __init__(
        self,
        var_weight: float = 0.5,
        min_var: float = 10.0,    # 原始空间真实方差~30，min_var=10 对应 std≥3.16
        mse_weight: float = 0.05, # 原始空间 MSE~30，乘 0.05 后≈1.5，与 CCC 量级匹配
    ) -> None:
        super().__init__()
        self.var_weight = var_weight
        self.min_var = min_var
        self.mse_weight = mse_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.float().reshape(-1)
        y_true = y_true.float().reshape(-1)
        pred_mean  = y_pred.mean()
        true_mean  = y_true.mean()
        pred_var   = ((y_pred - pred_mean) ** 2).mean()
        true_var   = ((y_true - true_mean) ** 2).mean()
        covariance = ((y_pred - pred_mean) * (y_true - true_mean)).mean()
        denom      = pred_var + true_var + (pred_mean - true_mean) ** 2 + 1e-8
        ccc_val    = 2.0 * covariance / denom
        ccc_loss   = 1.0 - ccc_val

        var_penalty = F.relu(self.min_var - pred_var)
        mse = F.mse_loss(y_pred, y_true)

        return ccc_loss + self.var_weight * var_penalty + self.mse_weight * mse


# =============================================================================
#  Ordinal BCE Loss + 反解码 (方案 A v1, 2026-05-15)
# =============================================================================
class OrdinalBCELoss(nn.Module):
    """
    把 PHQ-9 当 ordinal 多任务二分类训练。

    给定 K 个阈值 [t_1, ..., t_K]，训练 K 个二元分类器：
      head_k 学习 P(PHQ >= t_k)

    Loss = BCEWithLogits(logits[:, k], (phq >= t_k).float()) for each k
    每个 head 独立 pos_weight 处理长尾，训练前从训练集统计后传入。

    head_mask: [K] 0/1 buffer。某个 head 在训练集中正样本太少时（< mask_min_pos），
    传 0 静态屏蔽该 head 的 loss，避免噪声主导训练。
    """
    def __init__(
        self,
        thresholds: list[float],
        pos_weight: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.thresholds_list = list(thresholds)
        self.register_buffer(
            "thresholds_t",
            torch.tensor(thresholds, dtype=torch.float32),
        )
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None
        if head_mask is not None:
            self.register_buffer("head_mask", head_mask.float())
        else:
            self.head_mask = None

    def forward(self, logits: torch.Tensor, phq: torch.Tensor) -> torch.Tensor:
        # logits: [B, K], phq: [B]
        target = (phq.unsqueeze(-1) >= self.thresholds_t.to(phq.device)).float()
        bce = F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
            reduction="none",
        )  # [B, K]
        if self.head_mask is not None:
            mask = self.head_mask.to(logits.device)  # [K]
            bce = bce * mask
            denom = mask.sum().clamp(min=1.0) * float(logits.shape[0])
            return bce.sum() / denom
        return bce.mean()


def compute_ordinal_pos_weight(
    train_phq: torch.Tensor,
    thresholds: list[float],
    max_clamp: float = 5.0,
    min_clamp: float = 0.5,
) -> torch.Tensor:
    """
    从训练集 PHQ 标签计算每个阈值的 pos_weight = neg_count / pos_count，
    clamp 到 [min_clamp, max_clamp]：
      max_clamp 防止 0-positive 类别梯度爆炸；
      min_clamp 避免简单 head（pos > neg 时 ratio<<1）被弱化。
    返回 shape [K] 的 tensor。
    """
    thresholds_t = torch.tensor(thresholds, dtype=torch.float32)
    target = (train_phq.unsqueeze(-1) >= thresholds_t).float()   # [N, K]
    pos = target.sum(dim=0).clamp(min=1.0)                        # [K]，避免除 0
    neg = target.shape[0] - pos                                   # [K]
    return (neg / pos).clamp(min=min_clamp, max=max_clamp)


def compute_active_head_mask(
    train_phq: torch.Tensor,
    thresholds: list[float],
    min_pos: int = 0,
) -> torch.Tensor:
    """
    Per-fold 静态 head mask：训练集中 (PHQ>=t_k) 正样本数 < min_pos 的 head → 0。
    min_pos<=0 时返回全 1（不 mask）。
    """
    if min_pos <= 0:
        return torch.ones(len(thresholds), dtype=torch.float32)
    phq_np = train_phq.detach().cpu().numpy() if hasattr(train_phq, "detach") else np.asarray(train_phq)
    mask = [1.0 if int((phq_np >= t).sum()) >= min_pos else 0.0 for t in thresholds]
    return torch.tensor(mask, dtype=torch.float32)


def compute_empirical_midpoints(
    train_phq: torch.Tensor,
    thresholds: list[float],
    high_value: float = 27.0,
) -> list[float]:
    """
    从训练集 PHQ 算每个 bin 的 empirical 均值，作为 decode 时的代表值。
    bin 划分：[0, t_1), [t_1, t_2), ..., [t_{K-1}, t_K), [t_K, high_value]
    空 bin 退回几何中点，与原 decode 公式一致。
    """
    K = len(thresholds)
    phq_np = train_phq.detach().cpu().numpy() if hasattr(train_phq, "detach") else np.asarray(train_phq)
    edges = [0.0] + list(thresholds) + [high_value + 1e-3]
    midpoints: list[float] = []
    for k in range(K + 1):
        lo, hi = edges[k], edges[k + 1]
        m = (phq_np >= lo) & (phq_np < hi)
        if int(m.sum()) > 0:
            midpoints.append(float(phq_np[m].mean()))
            continue
        if k == 0:
            midpoints.append(thresholds[0] / 2.0)
        elif k == K:
            midpoints.append((thresholds[-1] + high_value) / 2.0)
        else:
            midpoints.append((thresholds[k - 1] + thresholds[k]) / 2.0)
    return midpoints


def decode_ordinal_to_phq(
    logits: torch.Tensor,
    thresholds: list[float],
    enforce_monotonic: bool = True,
    midpoints: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ordinal logits → 期望 PHQ + 5 档概率

    步骤：
      1. sigmoid 得到 K 个 P(PHQ >= t_k)
      2. （可选）cummin 强制单调递减：P(>=5) >= P(>=10) >= ... >= P(>=20)
      3. 差分得到 K+1 档区间概率：
         bin_0 = [0, t_1)        : 1 - P(>=t_1)
         bin_k = [t_{k-1}, t_k)  : P(>=t_{k-1}) - P(>=t_k)   (中间档)
         bin_K = [t_K, 27]       : P(>=t_K)
      4. 期望 PHQ = sum(bin_prob * bin_midpoint)
         中点：bin_0 = t_1 / 2; bin_k = (t_{k-1}+t_k)/2; bin_K = (t_K + 27) / 2

    返回:
        phq_pred: [B]      期望 PHQ-9 值
        bin_probs: [B, K+1] 5 档概率（K=4 时为 5 档）
    """
    K = len(thresholds)
    P_ge = torch.sigmoid(logits)                                  # [B, K]
    if enforce_monotonic:
        P_ge = torch.cummin(P_ge, dim=-1).values

    # 差分得到 K+1 档区间概率
    # P_left  = [1, P_ge[:,0], P_ge[:,1], ..., P_ge[:,K-1]]   shape [B, K+1]
    # P_right = [P_ge[:,0], P_ge[:,1], ..., P_ge[:,K-1], 0]   shape [B, K+1]
    # bin_probs = P_left - P_right
    ones = torch.ones(P_ge.shape[0], 1, device=P_ge.device, dtype=P_ge.dtype)
    zeros = torch.zeros_like(ones)
    P_left = torch.cat([ones, P_ge], dim=-1)
    P_right = torch.cat([P_ge, zeros], dim=-1)
    bin_probs = (P_left - P_right).clamp(min=0.0)                 # [B, K+1]
    bin_probs = bin_probs / bin_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    # 优先使用调用方传入的 midpoints（per-fold empirical bin mean），
    # 未传则按几何中点公式 (t_{k-1} + t_k) / 2，端点用 0/27 fallback
    if midpoints is None:
        midpoints = []
        for k in range(K + 1):
            if k == 0:
                mid = thresholds[0] / 2.0
            elif k == K:
                mid = (thresholds[-1] + 27.0) / 2.0
            else:
                mid = (thresholds[k-1] + thresholds[k]) / 2.0
            midpoints.append(mid)
    mid_t = torch.tensor(midpoints, device=P_ge.device, dtype=P_ge.dtype)

    phq_pred = (bin_probs * mid_t).sum(dim=-1)                    # [B]
    return phq_pred, bin_probs


# =============================================================================
#  Multitask Loss = ordinal BCE + binary CE + ternary CE + SmoothL1 + consistency
# =============================================================================
class MultitaskLoss(nn.Module):
    """
    Route 3 (multitask) 专用：组合 5 项 loss，对应 model 输出的 dict
        {ord_logits[B,K], bin_logits[B,2], ter_logits[B,3]}

    consistency 用 thresholds.index(boundary) 动态查 idx：
      - 若 bin_boundary  ∈ thresholds   → MSE(P(>=bin),  bin[1])  +  MSE(P(>=bin),  ter[1]+ter[2])
      - 若 ter_boundaries[1] ∈ thresholds → MSE(P(>=10),  ter[2])
      - 始终保留：MSE(bin[1], ter[1]+ter[2])
      列表里没有的阈值对应项自动跳过。

    SmoothL1(decoded_phq, true_phq) 由 phq_warmup_epochs 闸控（避免前期 ord head 未稳定时拉飞）。
    """
    def __init__(
        self,
        thresholds: list[float],
        pos_weight: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        midpoints: list[float] | None = None,
        loss_weights: tuple[float, float, float, float, float] = (1.0, 0.15, 0.30, 0.05, 0.10),
        bin_boundary: float = 5.0,
        ter_boundaries: tuple[float, float] = (5.0, 10.0),
        phq_warmup_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.thresholds_list = list(thresholds)
        self.ord_loss = OrdinalBCELoss(thresholds, pos_weight=pos_weight, head_mask=head_mask)
        self.bin_loss = nn.CrossEntropyLoss()
        self.ter_loss = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.weights = tuple(loss_weights)
        self.bin_boundary = float(bin_boundary)
        self.ter_boundaries = (float(ter_boundaries[0]), float(ter_boundaries[1]))
        self.midpoints = list(midpoints) if midpoints is not None else None
        self.phq_warmup_epochs = int(phq_warmup_epochs)
        # 动态查 consistency 索引（None = 该项跳过）
        self.idx_ge_bin = self._find(self.bin_boundary)
        self.idx_ge_ter1 = self._find(self.ter_boundaries[0])
        self.idx_ge_ter2 = self._find(self.ter_boundaries[1])
        # 当前 epoch（外部每个 epoch 起始时 set_epoch 一次）
        self.current_epoch = 0

    def _find(self, val: float) -> int | None:
        for i, t in enumerate(self.thresholds_list):
            if abs(float(t) - val) < 1e-6:
                return i
        return None

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        phq: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ord_logits = outputs["ord_logits"]
        bin_logits = outputs["bin_logits"]
        ter_logits = outputs["ter_logits"]
        device = ord_logits.device
        phq = phq.float()

        bin_target = (phq >= self.bin_boundary).long()
        ter_target = (
            (phq >= self.ter_boundaries[0]).long()
            + (phq >= self.ter_boundaries[1]).long()
        )

        loss_ord = self.ord_loss(ord_logits, phq)
        loss_bin = self.bin_loss(bin_logits, bin_target)
        loss_ter = self.ter_loss(ter_logits, ter_target)

        if self.current_epoch >= self.phq_warmup_epochs and self.weights[3] > 0:
            phq_decoded, _ = decode_ordinal_to_phq(
                ord_logits, self.thresholds_list,
                enforce_monotonic=True, midpoints=self.midpoints,
            )
            loss_phq = self.smooth_l1(phq_decoded, phq)
        else:
            loss_phq = torch.zeros((), device=device)

        loss_cons = self._consistency(ord_logits, bin_logits, ter_logits)

        w_ord, w_bin, w_ter, w_phq, w_cons = self.weights
        total = (
            w_ord * loss_ord
            + w_bin * loss_bin
            + w_ter * loss_ter
            + w_phq * loss_phq
            + w_cons * loss_cons
        )
        parts = {
            "ord":  float(loss_ord.item()),
            "bin":  float(loss_bin.item()),
            "ter":  float(loss_ter.item()),
            "phq":  float(loss_phq.item()),
            "cons": float(loss_cons.item()),
        }
        return total, parts

    def _consistency(
        self,
        ord_logits: torch.Tensor,
        bin_logits: torch.Tensor,
        ter_logits: torch.Tensor,
    ) -> torch.Tensor:
        ord_sig = torch.cummin(torch.sigmoid(ord_logits), dim=-1).values  # [B, K]
        bin_prob = F.softmax(bin_logits, dim=-1)                          # [B, 2]
        ter_prob = F.softmax(ter_logits, dim=-1)                          # [B, 3]
        terms = []
        if self.idx_ge_bin is not None:
            P_ord_b = ord_sig[:, self.idx_ge_bin]
            terms.append(F.mse_loss(P_ord_b, bin_prob[:, 1]))
            terms.append(F.mse_loss(P_ord_b, ter_prob[:, 1] + ter_prob[:, 2]))
        if self.idx_ge_ter2 is not None:
            P_ord_2 = ord_sig[:, self.idx_ge_ter2]
            terms.append(F.mse_loss(P_ord_2, ter_prob[:, 2]))
        terms.append(F.mse_loss(bin_prob[:, 1], ter_prob[:, 1] + ter_prob[:, 2]))
        if not terms:
            return torch.zeros((), device=ord_logits.device)
        return torch.stack(terms).mean()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = {
        "acc": safe_float(accuracy_score(y_true, y_pred)),
        "f1": safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": safe_float(cohen_kappa_score(y_true, y_pred)),
        "ccc": ccc(y_true, y_pred),
        "rmse": safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": safe_float(mean_absolute_error(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    metrics["selection_score"] = metrics["f1"]
    return metrics


# =============================================================================
#  修复 #2 (Bug 中间层): regression_metrics 同时输出两套指标
# =============================================================================
def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    phq_log1p: bool = True,
) -> dict[str, Any]:
    """
    计算回归指标，同时输出 log1p 空间和原始 PHQ-9 空间两套指标。

    【修复原因】
    原来只在 log1p 空间计算 rmse/mae，导致 val_rmse=0.578 看起来"还行"，
    实际上隐藏了模型完全没有区分度的问题（预测值集中在均值附近）。

    【修复方式】
    当 phq_log1p=True 时，额外对 y_true/y_pred 做 expm1() 还原，
    计算原始 PHQ-9 空间（0~27）的 rmse/mae/ccc，并加入 metrics 字典。

    新增字段：
      phq9_rmse       : 原始 PHQ-9 空间的 RMSE（直觉上更好理解，e.g. 误差 3 分）
      phq9_mae        : 原始 PHQ-9 空间的 MAE
      phq9_ccc        : 原始 PHQ-9 空间的 CCC（与提交评测一致）
      phq9_pred_mean  : 预测值均值（还原后），用于快速判断是否退化为均值预测
      phq9_pred_std   : 预测值标准差（还原后），接近 0 说明退化
      phq9_true_mean  : 真实值均值（还原后），与 phq9_pred_mean 对比
    """
    metrics = {
        # log1p 空间（训练时直接优化的目标）
        "ccc":  ccc(y_true, y_pred),
        "rmse": safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  safe_float(mean_absolute_error(y_true, y_pred)),
        "r2":   safe_float(r2_score(y_true, y_pred)),
    }
    metrics["selection_score"] = metrics["ccc"]

    # 始终输出 phq9_* 字段；区别仅在是否做 expm1 反变换
    if phq_log1p:
        phq_true_orig = np.clip(np.expm1(y_true), 0, 27)
        phq_pred_orig = np.clip(np.expm1(y_pred), 0, 27)
    else:
        # 【档位 B】raw 空间：值已是 [0,27]，只需 clip 不做 expm1
        phq_true_orig = np.clip(y_true, 0, 27)
        phq_pred_orig = np.clip(y_pred, 0, 27)
    metrics["phq9_rmse"]      = safe_float(np.sqrt(mean_squared_error(phq_true_orig, phq_pred_orig)))
    metrics["phq9_mae"]       = safe_float(mean_absolute_error(phq_true_orig, phq_pred_orig))
    metrics["phq9_ccc"]       = ccc(phq_true_orig, phq_pred_orig)
    metrics["phq9_pred_mean"] = float(phq_pred_orig.mean())
    metrics["phq9_pred_std"]  = float(phq_pred_orig.std())
    metrics["phq9_true_mean"] = float(phq_true_orig.mean())

    return metrics


def joint_regression_metrics(
    class_true: np.ndarray,
    class_pred: np.ndarray,
    phq_true: np.ndarray,
    phq_pred: np.ndarray,
) -> dict[str, Any]:
    metrics = classification_metrics(class_true, class_pred)
    reg_metrics = regression_metrics(phq_true, phq_pred)
    metrics["ccc"] = reg_metrics["ccc"]
    metrics["rmse"] = reg_metrics["rmse"]
    metrics["mae"] = reg_metrics["mae"]
    metrics["r2"] = reg_metrics["r2"]
    metrics["selection_score"] = metrics["f1"]
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Any,
    device: torch.device,
    task: str,
) -> dict[str, Any]:
    # is_joint_regression: 旧模式，分类+回归头共存（tuple criterion）
    # is_ordinal_regression: 新模式 (方案 A v1)，回归任务用 ordinal BCE，criterion 是 OrdinalBCELoss
    # is_pure_regression:  原"新"模式，task=regression 时只有 1 个标量回归头 + CCCLoss
    is_joint_regression = isinstance(criterion, (tuple, list))
    is_multitask = isinstance(criterion, MultitaskLoss)
    is_ordinal_regression = isinstance(criterion, OrdinalBCELoss) or is_multitask
    from dataset import REGRESSION_TASK
    is_pure_regression = (
        (task == REGRESSION_TASK) and not is_joint_regression and not is_ordinal_regression
    )
    # ordinal 模式下从 criterion 拿阈值，反解码用
    if is_multitask:
        ordinal_thresholds = criterion.thresholds_list
        ordinal_midpoints = criterion.midpoints
    elif isinstance(criterion, OrdinalBCELoss):
        ordinal_thresholds = criterion.thresholds_list
        ordinal_midpoints = None  # 可由调用方扩展（评估时几何中点已可用）
    else:
        ordinal_thresholds = None
        ordinal_midpoints = None

    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []
    all_ids: list[int] = []
    all_phq_preds: list[float] = []
    all_phq_labels: list[float] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            outputs = model(
                audio=batch["audio"].to(device) if "audio" in batch else None,
                video=batch["video"].to(device) if "video" in batch else None,
                gait=batch["gait"].to(device) if "gait" in batch else None,
                personality=batch["personality"].to(device),
                pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
            )
            if is_multitask:
                # Multitask (Route 3)：模型输出 dict，criterion 返回 (loss, parts)
                # selection 仍按 phq9_ccc，bin/ter 头的诊断指标在 train.py 另算
                phq9 = batch["phq9"].to(device)
                ord_logits = outputs["ord_logits"]
                loss, _parts = criterion(outputs, phq9)
                phq_pred, _ = decode_ordinal_to_phq(
                    ord_logits, ordinal_thresholds,
                    enforce_monotonic=True, midpoints=ordinal_midpoints,
                )
                batch_phq_preds = phq_pred.cpu().float().numpy().tolist()
                batch_phq_labels = phq9.cpu().float().numpy().tolist()
                batch_preds = []
                batch_labels = labels.cpu().numpy().tolist()
                total_reg_loss += float(loss.item()) * len(batch_labels)
            elif is_ordinal_regression:
                # Ordinal 回归模式：模型输出 [B, K] 个 logits，BCE loss 训练
                phq9 = batch["phq9"].to(device)
                ord_logits = outputs                            # [B, K]
                loss = criterion(ord_logits, phq9)
                phq_pred, _ = decode_ordinal_to_phq(
                    ord_logits, ordinal_thresholds,
                    enforce_monotonic=True, midpoints=ordinal_midpoints,
                )
                batch_phq_preds = phq_pred.cpu().float().numpy().tolist()
                batch_phq_labels = phq9.cpu().float().numpy().tolist()
                batch_preds = []
                batch_labels = labels.cpu().numpy().tolist()
                total_reg_loss += float(loss.item()) * len(batch_labels)
            elif is_pure_regression:
                # 纯回归模式（旧）：模型直接输出 1 个标量，CCCLoss 训练
                phq9 = batch["phq9"].to(device)
                reg_out = outputs.squeeze(-1)  # [batch_size]
                loss = criterion(reg_out, phq9)
                batch_phq_preds = reg_out.cpu().float().numpy().tolist()
                batch_phq_labels = phq9.cpu().float().numpy().tolist()
                # 用回归值阈值化得到分类预测（用于打印参考，不影响 CCC 选模）
                batch_preds = []
                batch_labels = labels.cpu().numpy().tolist()
                total_reg_loss += float(loss.item()) * len(batch_labels)
            elif is_joint_regression:
                # 联合模式（旧）：分类头 + 回归头共存
                criterion_cls, criterion_reg = criterion
                phq9 = batch["phq9"].to(device)
                logits, reg_out = outputs
                loss_cls = criterion_cls(logits, labels)
                loss_reg = criterion_reg(reg_out, phq9)
                loss = loss_cls + loss_reg
                batch_preds = logits.argmax(dim=-1).cpu().numpy().tolist()
                batch_labels = labels.cpu().numpy().tolist()
                batch_phq_preds = reg_out.cpu().numpy().tolist()
                batch_phq_labels = phq9.cpu().numpy().tolist()
                total_cls_loss += float(loss_cls.item()) * len(batch_labels)
                total_reg_loss += float(loss_reg.item()) * len(batch_labels)
            else:
                # 纯分类模式
                logits = outputs
                loss = criterion(logits, labels)
                batch_preds = logits.argmax(dim=-1).cpu().numpy().tolist()
                batch_labels = labels.cpu().numpy().tolist()

            total_loss += float(loss.item()) * len(batch_labels)
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            all_ids.extend(batch["pid"].cpu().numpy().tolist())
            if is_pure_regression or is_joint_regression or is_ordinal_regression:
                all_phq_preds.extend(batch_phq_preds)
                all_phq_labels.extend(batch_phq_labels)

    if is_ordinal_regression:
        phq_true = np.asarray(all_phq_labels, dtype=np.float64)
        phq_pred = np.asarray(all_phq_preds, dtype=np.float64)
        # ordinal 反解码后已是 raw PHQ 空间，phq_log1p=False
        metrics = regression_metrics(phq_true, phq_pred, phq_log1p=False)
        metrics.setdefault("f1", 0.0)
        metrics.setdefault("acc", 0.0)
        metrics.setdefault("kappa", 0.0)
        metrics["phq_true"] = phq_true.tolist()
        metrics["phq_pred"] = phq_pred.tolist()
        metrics["y_true"] = phq_true.tolist()
        metrics["y_pred"] = phq_pred.tolist()
        metrics["reg_loss"] = safe_float(total_reg_loss / max(1, len(all_phq_labels)))
    elif is_pure_regression:
        phq_true = np.asarray(all_phq_labels, dtype=np.float64)
        phq_pred = np.asarray(all_phq_preds, dtype=np.float64)
        # 【档位 B 修复】2026-05-13: 自动检测目标空间
        # log1p 空间下真实值不会超过 log1p(27)≈3.33；如果观察到 > 3.5 说明已是 raw 空间
        # 这样旧 checkpoint（log1p）和新训练（raw）都能得到正确的 [PHQ-9] 诊断指标
        _is_log1p = bool(np.max(phq_true) <= 3.5)
        metrics = regression_metrics(phq_true, phq_pred, phq_log1p=_is_log1p)
        metrics.setdefault("f1", 0.0)
        metrics.setdefault("acc", 0.0)
        metrics.setdefault("kappa", 0.0)
        metrics["phq_true"] = phq_true.tolist()
        metrics["phq_pred"] = phq_pred.tolist()
        metrics["y_true"] = phq_true.tolist()
        metrics["y_pred"] = phq_pred.tolist()
        metrics["reg_loss"] = safe_float(total_reg_loss / max(1, len(all_phq_labels)))
    elif is_joint_regression:
        y_true = np.asarray(all_labels, dtype=np.int64)
        y_pred = np.asarray(all_preds, dtype=np.int64)
        phq_true = np.asarray(all_phq_labels, dtype=np.float64)
        phq_pred = np.asarray(all_phq_preds, dtype=np.float64)
        metrics = joint_regression_metrics(y_true, y_pred, phq_true, phq_pred)
        metrics["cls_loss"] = safe_float(total_cls_loss / max(1, len(all_labels)))
        metrics["reg_loss"] = safe_float(total_reg_loss / max(1, len(all_labels)))
        metrics["class_true"] = y_true.tolist()
        metrics["class_pred"] = y_pred.tolist()
        metrics["phq_true"] = phq_true.tolist()
        metrics["phq_pred"] = phq_pred.tolist()
        metrics["y_true"] = phq_true.tolist()
        metrics["y_pred"] = phq_pred.tolist()
    else:
        y_true = np.asarray(all_labels, dtype=np.int64)
        y_pred = np.asarray(all_preds, dtype=np.int64)
        metrics = classification_metrics(y_true, y_pred)
        metrics["y_true"] = y_true.tolist()
        metrics["y_pred"] = y_pred.tolist()

    if task == REGRESSION_TASK:
        metrics["selection_score"] = safe_float(metrics.get("ccc", 0.0))
    else:
        metrics["selection_score"] = safe_float(metrics.get("f1", metrics.get("ccc", 0.0)))

    metrics["loss"] = safe_float(total_loss / max(1, len(all_ids)))
    metrics["ids"] = all_ids
    return metrics
