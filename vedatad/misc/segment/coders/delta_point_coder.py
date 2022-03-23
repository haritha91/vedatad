# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import torch

from vedacore.misc import registry
from .base_segment_coder import BaseSegmentCoder


@registry.register_module('segment_coder')
class DeltaPointCoder(BaseSegmentCoder):
    """Delta Point coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes segment (start, end) into delta (d_center,)
    and decodes delta (d_center, ) back to original segment
    (start, end).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self, target_means=(0., 0.), target_stds=(1., 1.)):
        super(BaseSegmentCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, segments, gt_segments):
        """Get segment regression transformation deltas that can be used to
        transform the ``segments`` into the ``gt_points``.

        Args:
            segments (torch.Tensor): Source segments, e.g., object proposals.
            gt_segments (torch.Tensor): Target of the transformation, e.g.,
                ground-truth segments.

        Returns:
            torch.Tensor: transformation point deltas
        """
        print('delta_point encoder')
        assert segments.size(0) == gt_segments.size(0)
        assert segments.size(-1) == gt_segments.size(-1) == 2
        encoded_segments = segment2delta(segments, gt_segments, self.means,
                                         self.stds)
        return encoded_segments

    def decode(self, segments, pred_segments, max_t=None):
        """Apply transformation `pred_segments` to `segments`.

        Args:
            segments (torch.Tensor): Basic segments.
            pred_segments (torch.Tensor): Encoded segments with shape
            max_t (int, optional): Maximum time of segments.
                Defaults to None.

        Returns:
            torch.Tensor: Decoded segments.
        """
        print('delta_point decoder')
        # print('decoder pred_segments - ', pred_segments.shape)
        # print('decoder segments - ', segments.shape)
        # print('pred_segments - ', pred_segments)
        # print('segments - ', segments)
        assert pred_segments.size(0) == segments.size(0)
        decoded_segments = delta2point(segments, pred_segments, self.means,
                                         self.stds, max_t)

        return decoded_segments

#encode
def segment2delta(proposals, gt, means=(0., 0.), stds=(1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of center, interval of proposals w.r.t ground
    truth segments to get regression target.
    This is the inverse function of :func:`delta2point`.

    Args:
        proposals (Tensor): Segments to be transformed, shape (N, ..., 2)
        gt (Tensor): Gt segments to be used as base, shape (N, ..., 2)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, ), where columns represent d_center,
    """
    assert proposals.size() == gt.size()

    # ## delta point approach
    # proposals = proposals.float()
    # gt = gt.float()

    # p_center = (proposals[..., 0] + proposals[..., 1]) * 0.5
    # p_interval = proposals[..., 1] - proposals[..., 0]

    # g_center = (gt[..., 0] + gt[..., 1]) * 0.5

    # d_center = (g_center - p_center) / p_interval
    # deltas = torch.stack([d_center], dim=-1)

    # means = deltas.new_tensor(means).unsqueeze(0)
    # stds = deltas.new_tensor(stds).unsqueeze(0)
    # deltas = deltas.sub_(means).div_(stds)
    # ######

    proposals = proposals.float()
    gt = gt.float()

    p_center = (proposals[..., 0] + proposals[..., 1]) * 0.5
    p_interval = proposals[..., 1] - proposals[..., 0]

    g_center = (gt[..., 0] + gt[..., 1]) * 0.5
    g_interval = gt[..., 1] - gt[..., 0]

    d_center = (g_center - p_center) / p_interval
    d_interval = torch.log(g_interval / p_interval)
    deltas = torch.stack([d_center, d_interval], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)


    return deltas

#decode
def delta2point(rois, deltas, means=(0.), stds=(1.), max_t=None):
    """Apply deltas to shift base points.

    Typically the rois are anchor or proposed segments and the deltas are
    network outputs used to shift those segments.
    This is the inverse function of :func:`segment2delta`.

    Args:
        rois (Tensor): Segments to be transformed. Has shape (N, 1)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 1 * num_classes). Note N = num_anchors * T when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_t (int): Maximum time for segments. specifies T

    Returns:
        Tensor: Points with shape (N)

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  1.],
        >>>                      [ 0.,  1.],
        >>>                      [ 0.,  1.],
        >>>                      [ 5., 5.]])
        >>> deltas = torch.Tensor([[  0.],
        >>>                        [  1.],
        >>>                        [  0.],
        >>>                        [ 0.7]])
        >>> delta2point(rois, deltas, max_t=32)
        tensor([[0.0000],
                [0.1409],
                [0.0000],
                [5.0000]])
    """
    print('delta2point method')
    # print('deltas - ', deltas.shape)
    # print('rois - ', rois.shape)


    means = deltas.new_tensor(means).repeat(1, deltas.size(0) // 2)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(0) // 2)
    # denorm_deltas = deltas * stds + means
    denorm_deltas = deltas

    # print('means shape - ', means.shape)
    # print('stds shape - ', stds.shape)
    # print('denorm_deltas - shape - ', denorm_deltas.shape)

    # Use delta to shift the center of each roi
    point = rois +  denorm_deltas

    # print('point - ', point.shape)

    if max_t is not None:
        point = start.clamp(min=0, max=max_t)
    points = torch.stack([point], dim=-1).view_as(deltas) #view as is deltas in original

    # print('points - ', points.shape)
    return points
