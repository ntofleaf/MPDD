import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from .utils import weight_reduce_loss


def mse_loss(pred,
             label,
             weight=None,
             reduction='mean',
             avg_factor=None,
             class_weight=None,
             rmse=False):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    # loss = F.mse_loss(pred, label.unsqueeze(dim=1), reduction='none')
    loss = nn.MSELoss()(pred, label)
    if rmse:
        loss = loss.sqrt()
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@MODELS.register_module()
class RegressionLoss(nn.Module):
    """Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_soft=False,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None,
                 pos_weight=None,
                 rmse=False):
        super(RegressionLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_soft = use_soft
        self.rmse = rmse
        assert not (
                self.use_soft and self.use_sigmoid
        ), 'use_sigmoid and use_soft could not be set simultaneously'

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.pos_weight = pos_weight

        if self.use_sigmoid:
            self.reg_criterion = None
            raise NotImplementedError
        elif self.use_soft:
            self.reg_criterion = None
            raise NotImplementedError
        else:
            self.reg_criterion = mse_loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # only BCE loss has pos_weight
        if self.pos_weight is not None and self.use_sigmoid:
            pos_weight = cls_score.new_tensor(self.pos_weight)
            kwargs.update({'pos_weight': pos_weight})
        else:
            pos_weight = None

        loss_reg = self.loss_weight * self.reg_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            rmse=self.rmse,
            **kwargs)
        return loss_reg
