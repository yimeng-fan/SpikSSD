from typing import Dict, List
import torch
import torch.nn as nn
from torch import Tensor
from models.SSD_utils import init_weights


class SSDHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, bn=True, args=None):
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes, bn=bn)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors, bn=bn)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
        }

class SSDScoringHead(nn.Module):
    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def forward(self, x: List[Tensor]) -> Tensor:
        all_results = []

        for i, features in enumerate(x):
            results = self.module_list[i](features)

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


class SSDClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, bn=True):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(nn.Sequential(
                # nn.Dropout2d(p=0.2),s
                nn.ConstantPad2d(1, 0.),
                nn.BatchNorm2d(channels) if bn else nn.Identity(channels),
                nn.Conv2d(channels, num_classes * anchors, kernel_size=3, bias=False),
            ))
        cls_logits.apply(init_weights)
        super().__init__(cls_logits, num_classes)


class SSDRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], bn=True):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Sequential(
                # nn.Dropout2d(p=0.2),
                nn.ConstantPad2d(1, 0.),
                nn.BatchNorm2d(channels) if bn else nn.Identity(channels),
                nn.Conv2d(channels, 4 * anchors, kernel_size=3, bias=False),
            ))
        bbox_reg.apply(init_weights)
        super().__init__(bbox_reg, 4)
