# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import torch
import torch.nn as nn

from vedacore.misc import registry
from .utils import weighted_loss


@weighted_loss
def point_distance_loss(pred, target, eps=1e-7):
    r"""`Implementation of Point distance Loss: for Bounding Box Regression.
    Args:
        pred (Tensor): Predicted segments of format (start, end),
            shape (n, 2).
        target (Tensor): Corresponding gt segments, shape (n, 2).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    pred_center = (pred[:, 0] + pred[:, 1]) * 0.5
    target_center = (target[:, 0] + target[:, 1]) * 0.5
    distance = abs(target_center - pred_center)

    # enclose area
    enclose_start = torch.min(pred[:, 0], target[:, 0])
    enclose_end = torch.max(pred[:, 1], target[:, 1])
    enclose_interval = (enclose_end - enclose_start).clamp(min=0)
    c = enclose_interval + eps

    #normalized distance 
    loss = distance / c
    return loss


@registry.register_module('loss')
class PDLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(PDLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * point_distance_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
