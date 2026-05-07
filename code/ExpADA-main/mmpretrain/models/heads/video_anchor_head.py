# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule
import numpy as np
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class VideoAnchorHead(BaseModule):
    """Regression head.

    Args:
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 loss: dict = dict(type='RegressionLoss', loss_weight=1.0),
                 init_cfg: Optional[dict] = None):
        super(VideoAnchorHead, self).__init__(init_cfg=init_cfg)

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.reg_loss = loss
        self.wea_loss = nn.BCELoss()
        self.cls_loss = nn.BCELoss()

    # def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
    #     """The process before the final classification head.

    #     The input ``feats`` is a tuple of tensor, and each tensor is the
    #     feature of a backbone stage. In ``ClsHead``, we just obtain the feature
    #     of the last stage.
    #     """
    #     # The ClsHead doesn't have other module, just return after unpacking.
    #     return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        return feats

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        feats = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(feats, data_samples, **kwargs)
        return losses

    def _get_loss(self, feats: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        smp_on_cls_pred, cls_pred, reg_pred = feats
        # compute loss
        losses = dict()

        cls_target = torch.tensor(data_samples[0].dep_cls).float().cuda()
        cls_target_r = cls_target.repeat(cls_pred.size(0), 1).float()
        # cls_target = torch.unsqueeze(cls_target, 0)
        
        losses["wea_loss"] = self.wea_loss(
            smp_on_cls_pred, cls_target)
        losses["cls_loss"] = self.cls_loss(cls_pred, cls_target_r)
        # reg_pred = torch.tensor([[reg_pred]]).cuda()
        target = torch.unsqueeze(target, 1)
        target = target.repeat(reg_pred.size(0), 1).float()
        # target = target.float()
        losses["reg_loss"] = self.reg_loss(reg_pred, target, avg_factor=1, **kwargs)

        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[DataSample]]] = None
    ) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        feats = self(feats)
        _, cls_pred, reg_pred = feats

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_pred, reg_pred, data_samples)
        return predictions

    def _get_predictions(self, cls_pred, reg_pred, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        cls_pred = cls_pred.mean(dim=0)
        # pred_scores = torch.tensor([reg_pred]).cuda()
        # TODO: chech to use mean or sum !!!
        pred_scores = reg_pred
        pred_scores = pred_scores.mean(dim=0)
        data_samples[0].set_pred_score(pred_scores)
        data_samples[0].set_pred_label(cls_pred)
        return data_samples

        # if data_samples is None:
        #     data_samples = [None for _ in range(pred_scores.size(0))]

        # for data_sample, score in zip(data_samples, pred_scores):
        #     if data_sample is None:
        #         data_sample = DataSample()

        #     data_sample.set_pred_score(score)
        #     out_data_samples.append(data_sample)
        # return out_data_samples


class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(WeightedAsymmetricLoss, self).__init__()
        self.rate = [77, 22, 26, 25]
        self.weight_deno = np.array([1 / rate for rate in self.rate]).sum()
        self.weights = [(1 / rate) / self.weight_deno for rate in self.rate]
        self.weights = torch.tensor(self.weights, dtype=torch.float32)

    def forward(self, y_pred, y_true, epsilon=1e-8):
        """
        Compute binary cross-entropy loss manually.

        Args:
        y_pred (torch.Tensor): Tensor containing predicted probabilities.
        y_true (torch.Tensor): Tensor containing actual labels (0 or 1).
        epsilon (float): Small value to ensure numerical stability.

        Returns:
        torch.Tensor: The computed binary cross-entropy loss.
        """
        # rates = [77, 22, 26, 25]
        # weight_deno = np.array([ 1 / rate for rate in rates]).sum()
        # weights = [(1 / rate) / weight_deno for rate in rates]
        # weights = torch.tensor(weights, dtype=torch.float32)
        weights = self.weights.to(y_pred.device)
        # Ensure the predicted probabilities are clipped to avoid log(0)
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

        # Calculate the binary cross entropy
        loss = - y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
        loss = weights * loss
        # Return the mean of the loss across all observations
        return torch.mean(loss)