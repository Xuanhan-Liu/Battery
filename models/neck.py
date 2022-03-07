import torch
from torch import nn
from activation import Mish
import torch.nn.functional as F


class PSPModule(nn.Module):
    def __init__(self, in_c, pool_sizes, norm_layer='bn', activation='relu'):
        super(PSPModule, self).__init__()
        out_c = in_c / pool_sizes

        self.stages = nn.ModuleList(
            [self._make_stages(in_c, out_c, pool_size, norm_layer, activation) for pool_size in pool_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_c + (out_c * len(pool_sizes)), out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )

    def _make_stages(self, in_c, out_c, size, norm_layer, activation):
        pool = nn.AdaptiveAvgPool3d(output_size=size)
        conv = nn.Conv3d(in_c, out_c, kernel_size=1, bias=False)
        if norm_layer == 'bn':
            norm = nn.BatchNorm3d(out_c)
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'mish':
            act = Mish()
        return nn.Sequential(pool, conv, norm, act)

    def forward(self, x):
        h, w, d = x.size(2), x.size(3), x.size(4)
        pyramids = [x]
        pyramids.extend(
            [F.interpolate(stage(x), size=(h, w, d), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output
