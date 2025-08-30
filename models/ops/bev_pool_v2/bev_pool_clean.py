# Copyright (c) Phigent Robotics. All rights reserved.

import numpy as np
import torch

# Import fallback implementation for CPU compatibility
try:
    from . import bev_pool_v2_ext
except ImportError:
    print("Warning: Using CPU fallback for bev_pool_v2")
    from .bev_pool_fallback import bev_pool_v2, bev_pool_v2_ext, QuickCumsumCuda

__all__ = ['bev_pool_v2', 'TRTBEVPoolv2']

# Original QuickCumsumCuda class commented out - using fallback from bev_pool_fallback.py

def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    x = QuickCumsumCuda.apply(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts,
                              interval_lengths)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


class TRTBEVPoolv2(torch.autograd.Function):

    @staticmethod
    def symbolic(g,
                 depth,
                 feat,
                 ranks_depth,
                 ranks_feat,
                 ranks_bev,
                 interval_starts,
                 interval_lengths,
                 out_height=128,
                 out_width=128):
        """symbolic function for creating onnx op."""
        return g.op(
            'mmdeploy::bev_pool_v2',
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            out_height_i=out_height,
            out_width_i=out_width)

    @staticmethod
    def forward(g,
                depth,  # N,D,H,W
                feat,  # N,H,W,C
                ranks_depth,
                ranks_feat,
                ranks_bev,
                interval_starts,
                interval_lengths,
                out_height=128,
                out_width=128):
        """run forward."""
        feat = feat.unsqueeze(0)
        depth = depth.unsqueeze(0)
        bev_feat_shape = (depth.shape[0], 1, out_height, out_width,
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        bev_feat = bev_feat.squeeze(2)
        bev_feat = bev_feat.permute(0, 2, 3, 1)
        return bev_feat


def test_bev_pool_v2():
    depth = np.array([0.3, 0.4, 0.2, 0.1, 0.7, 0.6, 0.8, 0.9])
    depth = torch.from_numpy(depth).float()
    depth = depth.view(1, 1, 2, 2, 2).requires_grad_()
    feat = torch.ones(
        size=[1, 1, 2, 2, 2], dtype=torch.float,
        device='auto').requires_grad_()
    ranks_depth = torch.from_numpy(np.array([0, 4, 1, 6])).int()
    ranks_feat = torch.from_numpy(np.array([0, 0, 1, 2])).int()
    ranks_bev = torch.from_numpy(np.array([0, 0, 1, 1])).int()

    kept = torch.ones(
        ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    if len(interval_starts) == 0:
        return None, None, None, None, None
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
    bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                           (1, 1, 2, 2, 2), interval_starts, interval_lengths)
    loss = torch.sum(bev_feat)
    loss.backward()
    assert loss == 4.4
    grad_depth = np.array([2., 2., 0., 0., 2., 0., 2., 0.])
    grad_depth = torch.from_numpy(grad_depth).float()
    grad_depth = grad_depth.view(1, 1, 2, 2, 2)
    assert depth.grad.allclose(grad_depth)
    grad_feat = np.array([1.0, 1.0, 0.4, 0.4, 0.8, 0.8, 0., 0.])
    grad_feat = torch.from_numpy(grad_feat).float().view(1, 1, 2, 2, 2)
    assert feat.grad.allclose(grad_feat)
