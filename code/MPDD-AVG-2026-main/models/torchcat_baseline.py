from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_temporal_encoder import HybridTemporalEncoder


class ModalityEncoder(nn.Module):
    """
    【档位 C 修复】2026-05-13
    旧版 forward 末尾 `self.norm(x.mean(dim=1))` 是回归塌缩的核心元凶：
      1. mean(dim=1) 把整段时序平均，5 秒哭泣信号被 60 秒平均稀释 12 倍
      2. LayerNorm 把每个样本归一化到 0 均值单位方差，
         直接抹掉"严重抑郁特征幅度大、轻症幅度小"这个区分信号

    现改为：
      1. mean + max 双池化拼接，保留"整体趋势 + 最强烈片段"
      2. pool_proj 把 2H 维投影回 H 维，保持下游接口不变
      3. 去掉 LayerNorm，让样本间的特征幅度差异保留下来
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.5,
        pre_dim: int | None = None,
    ) -> None:
        super().__init__()
        if pre_dim is not None and input_dim > pre_dim:
            self.pre_proj = nn.Linear(input_dim, pre_dim)
            lstm_in = pre_dim
        else:
            self.pre_proj = None
            lstm_in = input_dim
        self.proj = nn.Linear(lstm_in, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # 新增：max+mean 拼接后的投影层，把 2H -> H 保持接口
        self.pool_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten_parameters() is required when using nn.DataParallel:
        # DataParallel spawns one thread per GPU and all threads call LSTM forward
        # simultaneously. Without this call, the LSTM's internal weight storage
        # (managed by cuDNN) is in a non-contiguous layout that causes a CUDA
        # memory deadlock across threads, hanging the process indefinitely.
        self.lstm.flatten_parameters()
        if self.pre_proj is not None:
            x = F.relu(self.pre_proj(x))
        x = F.relu(self.proj(x))
        x = self.dropout(x)
        x, _ = self.lstm(x)                         # [B, T, H]
        mean_pool = x.mean(dim=1)                   # 整体平均
        max_pool = x.max(dim=1).values              # 最强烈片段
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2H]
        return self.pool_proj(pooled)               # [B, H]


class PersonalityEncoder(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchcatBaseline(nn.Module):
    SUBTRACKS = {
        "A-V+P": ["audio", "video", "personality"],
        "A-V-G+P": ["audio", "video", "gait", "personality"],
        "G+P": ["gait", "personality"],
    }
    ENCODER_TYPES = {"bilstm_mean", "hybrid_attn"}

    MULTITASK_MODES = {"off", "ord_bin_ter"}

    def __init__(
        self,
        subtrack: str = "A-V-G+P",
        num_classes: int = 3,
        is_regression: bool = False,
        use_regression_head: bool = False,
        audio_dim: int = 64,
        video_dim: int = 1000,
        gait_dim: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        encoder_type: str = "bilstm_mean",
        regression_head_mode: str = "direct",
        ordinal_n_thresholds: int = 4,
        multitask_mode: str = "off",
    ) -> None:
        super().__init__()
        if subtrack not in self.SUBTRACKS:
            raise ValueError(f"Unknown subtrack: {subtrack}")
        if encoder_type not in self.ENCODER_TYPES:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        if regression_head_mode not in {"direct", "ordinal"}:
            raise ValueError(f"Unknown regression_head_mode: {regression_head_mode}")
        if multitask_mode not in self.MULTITASK_MODES:
            raise ValueError(f"Unknown multitask_mode: {multitask_mode}")

        self.subtrack = subtrack
        self.modalities = self.SUBTRACKS[subtrack]
        self.encoder_type = encoder_type
        self.is_regression = is_regression
        self.use_regression_head = use_regression_head
        self.regression_head_mode = regression_head_mode
        self.ordinal_n_thresholds = ordinal_n_thresholds
        self.multitask_mode = multitask_mode

        if "audio" in self.modalities:
            pre_audio = 128 if audio_dim > 128 else None
            self.audio_enc = (
                HybridTemporalEncoder(audio_dim, hidden_dim, dropout, pre_dim=pre_audio)
                if encoder_type == "hybrid_attn"
                else ModalityEncoder(audio_dim, hidden_dim, dropout, pre_dim=pre_audio)
            )
        if "video" in self.modalities:
            pre_video = 128 if video_dim > 128 else None
            self.video_enc = (
                HybridTemporalEncoder(video_dim, hidden_dim, dropout, pre_dim=pre_video)
                if encoder_type == "hybrid_attn"
                else ModalityEncoder(video_dim, hidden_dim, dropout, pre_dim=pre_video)
            )
        if "gait" in self.modalities:
            self.gait_enc = ModalityEncoder(gait_dim, hidden_dim, dropout)
        if "personality" in self.modalities:
            self.pers_enc = PersonalityEncoder(1024, hidden_dim, dropout)

        fused_dim = hidden_dim * len(self.modalities)
        self.fused_dim = fused_dim
        # 决定 classifier 输出维度：
        #   multitask=ord_bin_ter → 三个独立 head：ord_head[K], bin_head[2], ter_head[3]
        #     （forward 返回 dict，不走 self.classifier）
        #   ordinal 回归模式 → ordinal_n_thresholds 个 logits
        #   direct 回归模式  → 1 个标量
        #   分类任务         → num_classes 个 logits
        if multitask_mode == "ord_bin_ter":
            self.ord_head = self._make_head(fused_dim, hidden_dim, dropout, ordinal_n_thresholds)
            self.bin_head = self._make_head(fused_dim, hidden_dim, dropout, 2)
            self.ter_head = self._make_head(fused_dim, hidden_dim, dropout, 3)
        else:
            if is_regression and regression_head_mode == "ordinal":
                classifier_out = ordinal_n_thresholds
            elif is_regression:
                classifier_out = 1
            else:
                classifier_out = num_classes
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, classifier_out),
            )
            if use_regression_head:
                self.regressor = nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )

    @staticmethod
    def _make_head(in_dim: int, hidden: int, dropout: float, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    @staticmethod
    def _masked_average_sequences(x: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pair_mask is None:
            return x.mean(dim=1)
        weights = pair_mask.unsqueeze(-1).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    @staticmethod
    def _masked_average_features(x: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pair_mask is None:
            return x.mean(dim=1)
        weights = pair_mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def _encode_pairwise_sequences(
        self,
        x: torch.Tensor,
        encoder: nn.Module,
        pair_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, pair_count, seq_len, feat_dim = x.shape
        encoded = encoder(x.reshape(batch_size * pair_count, seq_len, feat_dim))
        encoded = encoded.reshape(batch_size, pair_count, -1)
        return self._masked_average_features(encoded, pair_mask)

    def forward(
        self,
        audio: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        gait: torch.Tensor | None = None,
        personality: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = []

        if "audio" in self.modalities:
            if self.encoder_type == "hybrid_attn":
                features.append(self._encode_pairwise_sequences(audio, self.audio_enc, pair_mask))
            else:
                features.append(self.audio_enc(self._masked_average_sequences(audio, pair_mask)))

        if "video" in self.modalities:
            if self.encoder_type == "hybrid_attn":
                features.append(self._encode_pairwise_sequences(video, self.video_enc, pair_mask))
            else:
                features.append(self.video_enc(self._masked_average_sequences(video, pair_mask)))

        if "gait" in self.modalities:
            features.append(self.gait_enc(gait))

        if "personality" in self.modalities:
            features.append(self.pers_enc(personality))

        fused = torch.cat(features, dim=-1)
        if self.multitask_mode == "ord_bin_ter":
            return {
                "ord_logits": self.ord_head(fused),
                "bin_logits": self.bin_head(fused),
                "ter_logits": self.ter_head(fused),
            }
        logits = self.classifier(fused)
        if self.use_regression_head:
            return logits, self.regressor(fused).squeeze(-1)
        return logits
