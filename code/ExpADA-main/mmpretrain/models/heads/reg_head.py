# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.models.losses.weighted_asymmetric_loss import weighted_asymmtric_bce_loss
# from mmpretrain.models.losses.weighted_asymmetric_loss import weighted_asymmtric_ce_loss


@MODELS.register_module()
class RegHead(BaseModule):
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
                 topk: Union[int, Tuple[int]] = (1, ),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        super(RegHead, self).__init__(init_cfg=init_cfg)

        self.topk = topk
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        self.cal_acc = cal_acc

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ClsHead``, we just obtain the feature
        of the last stage.
        """
        # The ClsHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The ClsHead doesn't have the final classification head,
        # just return the unpacked inputs.
        return pre_logits

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
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

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
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = cls_score

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]
        if pred_scores.dim() == 1:
            pred_scores = pred_scores.unsqueeze(1)
        for data_sample, score in zip(data_samples, pred_scores):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score)
            out_data_samples.append(data_sample)
        return out_data_samples


@MODELS.register_module()
class ClsRegHead(BaseModule):
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
                 topk: Union[int, Tuple[int]] = (1, ),
                 cal_acc: bool = True,
                 init_cfg: Optional[dict] = None):
        super(ClsRegHead, self).__init__(init_cfg=init_cfg)

        self.topk = topk
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        self.cal_acc = cal_acc
        if self.cal_acc:
            # self.bce_loss = nn.BCELoss()
            self.bce_loss = weighted_asymmtric_bce_loss

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ClsHead``, we just obtain the feature
        of the last stage.
        """
        # The ClsHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The ClsHead doesn't have the final classification head,
        # just return the unpacked inputs.
        return pre_logits

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
        reg_score, cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(reg_score, cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self,
                  reg_score: torch.Tensor,
                  cls_score: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])
        if 'gt_cls_label' not in data_samples[0]:
            raise ValueError('The `cls_label` is not in data samples.')

        cls_target = torch.stack(
            [torch.tensor(i.gt_cls_label).float() for i in data_samples]
        )
        cls_target = cls_target.to(cls_score.device)
        # compute loss
        losses = dict()
        reg_loss = self.loss_module(
            reg_score, target.unsqueeze(1), avg_factor=reg_score.size(0), **kwargs)
        losses['reg_loss'] = reg_loss
        cls_loss = self.bce_loss(cls_score, cls_target)
        losses['cls_loss'] = cls_loss
        # compute accuracy
        # if self.cal_acc:
        #     assert target.ndim == 1, 'If you enable batch augmentation ' \
        #         'like mixup during training, `cal_acc` is pointless.'
        #     acc = Accuracy.calculate(cls_score, cls_target.topk(1, dim=1)[1], topk=self.topk)
        #     losses.update(
        #         {f'accuracy_top-{k}': a
        #          for k, a in zip(self.topk, acc)})

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
        reg_score, cls_score = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(reg_score, cls_score, data_samples)
        return predictions

    def _get_predictions(self, reg_score, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = reg_score

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, re, cl in zip(data_samples, reg_score, cls_score):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(re)
            data_sample.set_pred_label(cl)
            out_data_samples.append(data_sample)
        return out_data_samples


@MODELS.register_module()
class AnchorClsRegHead(BaseModule):
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
                 topk: Union[int, Tuple[int]] = (1, ),
                 cal_acc: bool = True,
                 init_cfg: Optional[dict] = None):
        super(ClsRegHead, self).__init__(init_cfg=init_cfg)

        self.topk = topk
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        self.cal_acc = cal_acc
        if self.cal_acc:
            # self.bce_loss = nn.BCELoss()
            # self.bce_loss = weighted_asymmtric_bce_loss
            self.cls_loss = weighted_asymmtric_bce_loss

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ClsHead``, we just obtain the feature
        of the last stage.
        """
        # The ClsHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The ClsHead doesn't have the final classification head,
        # just return the unpacked inputs.
        return pre_logits

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
        reg_score, cls_score, com_embed = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(reg_score, cls_score, com_embed, data_samples, **kwargs)
        return losses

    def _get_loss(self,
                  reg_score: torch.Tensor,
                  cls_score: torch.Tensor,
                  com_embed: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])
        if 'gt_cls_label' not in data_samples[0]:
            raise ValueError('The `cls_label` is not in data samples.')

        cls_target = torch.stack(
            [torch.tensor(i.gt_cls_label).float() for i in data_samples]
        )
        cls_target = cls_target.to(cls_score.device)
        # compute loss
        losses = dict()
        reg_loss = self.loss_module(
            reg_score, target, avg_factor=reg_score.size(0), **kwargs)
        losses['reg_loss'] = reg_loss
        cls_loss = self.cls_loss(cls_score, cls_target)
        losses['cls_loss'] = cls_loss
        com_loss = compute_com_loss(com_embed, cls_target)
        losses.update(com_loss)
        # compute accuracy
        # if self.cal_acc:
        #     assert target.ndim == 1, 'If you enable batch augmentation ' \
        #         'like mixup during training, `cal_acc` is pointless.'
        #     acc = Accuracy.calculate(cls_score, cls_target.topk(1, dim=1)[1], topk=self.topk)
        #     losses.update(
        #         {f'accuracy_top-{k}': a
        #          for k, a in zip(self.topk, acc)})

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
        reg_score, cls_score, _ = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(reg_score, cls_score, data_samples)
        return predictions

    def _get_predictions(self, reg_score, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = reg_score

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, re, cl in zip(data_samples, reg_score, cls_score):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(re)
            data_sample.set_pred_label(cl)
            out_data_samples.append(data_sample)
        return out_data_samples


def compute_com_loss(com_embed, cls_target, margin=0.2):
    """Compute the loss of common embedding.

    Args:
        com_embed (torch.Tensor): The common embedding.
        cls_target (torch.Tensor): The target of classification.

    Returns:
        torch.Tensor: The computed loss.
    """
    losses = {}
    cls_label_num = cls_target[:, :4].sum(dim=0)
    slope = cls_target[:, 4:]
    cls_idx = torch.argmax(cls_target[:, :4], dim=1)
    if cls_label_num[0] > 0:  # cls 0 is not empty
        # using cls 0 as anchor
        # find the index of cls 0 in the batch
        cls_0_idx = torch.where(cls_idx == 0)[0]
        anc = com_embed[cls_0_idx[0]]
        anc_slope = slope[cls_0_idx[0]]
        if cls_label_num[0] > 1:
            pos = com_embed[cls_0_idx[1]]  # using cls 0 as anchor 
        else:
            pos = anc  # using as place holder for pos sample
        
        # search for n1, n2, n3
        if cls_label_num[1] > 0:  # TODO: cls 1 has 2 in the batch, compute its sim loss
            neg_1_idx = torch.where(cls_idx == 1)[0]
            neg1 = com_embed[neg_1_idx[0]]
            neg1_slope = slope[neg_1_idx[0]]
        else:
            neg1 = None
            neg1_slope = [1, 1]

        if cls_label_num[2] > 0:
            neg_2_idx = torch.where(cls_idx == 2)[0]
            neg2 = com_embed[neg_2_idx[0]]
            neg2_slope = slope[neg_2_idx[0]]
        else:
            neg2 = None
            neg2_slope = [1, 1]

        if cls_label_num[3] > 0:
            neg_3_idx = torch.where(cls_idx == 3)[0]
            neg3 = com_embed[neg_3_idx[0]]
            neg3_slope = slope[neg_3_idx[0]]
        else:
            neg3 = None
            neg3_slope = [1, 1]

        # dynamic weight 
        # dyw_12 = 2 / (neg1_slope + neg2_slope)  # TODO: use or not
        # dyw_23 = 2 / (neg2_slope + neg3_slope)
        # comput comparative loss
        if neg1 is not None:
            dyw_01 = 2 / (anc_slope[0] + neg1_slope[0])
            margin1 = margin * dyw_01
            losses["trip_loss_1"] = torch.max(
                torch.dist(anc, pos) - torch.dist(anc, neg1) + margin1, 
                torch.tensor(0).float().cuda(),
            )
        
        if neg2 is not None:
            # relative margin to eng1
            dyw_12 = 2 / (neg1_slope[1] + neg2_slope[0])
            margin2 = margin * dyw_12
            if neg1 is None:
                neg1 = pos
                # absolute margin to anchor
                margin2 = (1 + dyw_12) * margin
            losses["trip_loss_2"] = torch.max(
                torch.dist(anc, neg1) - torch.dist(anc, neg2) + margin2, 
                torch.tensor(0).float().cuda(),
            )

        if neg3 is not None:
            # relative margin to eng2
            dyw_23 = 2 / (neg2_slope[1] + neg3_slope[0])   
            margin3 = margin * dyw_23
            if neg2 is None:
                neg2 = pos
                # absolute margin to anchor
                margin3 = (2 + dyw_23) * margin
            losses["trip_loss_3"] = torch.max(
                torch.dist(anc, neg2) - torch.dist(anc, neg3) + margin3, 
                torch.tensor(0).float().cuda(),
            )
        # neg1, neg2 and neg3 are all None
        if cls_label_num[0] > 1:
            losses['contrastive_sim_loss_1'] = torch.dist(anc, pos)

    # if cls_label_num[3] > 0:  # cls 3 is not empty
    #     # using cls 3 as anchor
    #     # find the index of cls 3 in the batch
    #     cls_3_idx = torch.where(cls_idx == 3)[0]
    #     anc = com_embed[cls_3_idx[0]]
    #     neg3_slope = slope[cls_3_idx[0]]
    #     if cls_label_num[3] > 1:
    #         pos = com_embed[cls_3_idx[1]]
    #     else:
    #         pos = anc
        
    #     # search for n2, n1
    #     # search for n2
    #     if cls_label_num[2] > 0:
    #         neg_2_idx = torch.where(cls_idx == 2)[0]
    #         neg2 = com_embed[neg_2_idx[0]]
    #         neg2_slope = slope[neg_2_idx[0]]
    #     else:
    #         neg2 = None
    #         neg2_slope = [1, 1]

    #     # serch for n2
    #     if cls_label_num[1] > 0:
    #         neg_1_idx = torch.where(cls_idx == 1)[0]
    #         neg1 = com_embed[neg_1_idx[0]]
    #         neg1_slope = slope[neg_1_idx[0]]
    #     else:
    #         neg1 = None
    #         neg1_slope = [1, 1]
        
    #     # comput comparative loss
    #     if neg2 is not None:
    #         dyw_32 = 2 / (neg3_slope[0] + neg2_slope[1])
    #         margin1 = margin * dyw_32
    #         losses["trip_loss_4"] = torch.max(
    #             torch.dist(anc, pos) - torch.dist(anc, neg2) + margin1, 
    #             torch.tensor(0).float().cuda(),
    #         )
        
    #     if neg1 is not None:
    #         # relative margin to neg1
    #         dyw21 = 2 / (neg2_slope[0] + neg1_slope[1])
    #         margin2 = margin * dyw21
    #         if neg2 is None:
    #             neg2 = pos
    #             # absolute margin to anchor
    #             margin2 = (1 + dyw21) * margin
    #         losses["trip_loss_5"] = torch.max(
    #             torch.dist(anc, neg2) - torch.dist(anc, neg1) + margin2, 
    #             torch.tensor(0).float().cuda(),
    #         )
    #     if cls_label_num[3] > 1:
    #         losses['contrastive_sim_loss_4'] = torch.dist(anc, pos)

    if cls_label_num[1] > 0 and cls_label_num[2] > 0:  # using contrastive loss
        # using cls 1 as anchor
        # find the index of cls 1 in the batch
        cls_1_idx = torch.where(cls_idx == 1)[0]
        anc1 = com_embed[cls_1_idx[0]]
        neg1_slope = slope[cls_1_idx[0]]
        if cls_label_num[1] > 1:
            pos1 = com_embed[cls_1_idx[1]]
        else:
            pos1 = anc1
        losses['contrastive_sim_loss_2'] = torch.dist(anc1, pos1) * 10

        # using cls 2 as anchor
        # find the index of cls 2 in the batch
        cls_2_idx = torch.where(cls_idx == 2)[0]
        anc2 = com_embed[cls_2_idx[0]]
        neg2_slope = slope[cls_2_idx[0]]
        if cls_label_num[2] > 1:
            pos2 = com_embed[cls_2_idx[1]]
        else:
            pos2 = anc2
        losses['contrastive_sim_loss_3'] = torch.dist(anc2, pos2)

        dyw_12 = 10 * 2 / (neg1_slope[1] + neg2_slope[0])
        losses["contrastive_dis_loss"] = torch.max(
            dyw_12 * margin - torch.dist(pos1, pos2), 
            torch.tensor(0).float().cuda(),
        )
    
    # elif cls_label_num[1] >= 2:  # for case of cls labels just have 1
    #     cls_1_idx = torch.where(cls_idx == 1)[0]
    #     anc = com_embed[cls_1_idx[0]]
    #     pos = com_embed[cls_1_idx[1]]
    #     losses['contrastive_sim_loss_1'] = torch.dist(anc, pos)

    # else:  # for case of cls labels just have 2
    #     try:
    #         cls_2_idx = torch.where(cls_idx == 2)[0]
    #         anc = com_embed[cls_2_idx[0]]
    #         pos = com_embed[cls_2_idx[1]]
    #         losses['contrastive_sim_loss_2'] = torch.dist(anc, pos)
    #     except:
    #         pass
    return losses 