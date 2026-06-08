from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from proj4_code.segmentation.resnet import resnet50

class SimpleSegmentationNet(nn.Module):
    """
    ResNet backbone, with no increased dilation and no PPM, and a barebones
    classifier.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        deep_base: bool = True,
    ) -> None:
        """ """
        # super(SimpleSegmentationNet, self).__init__()
        super().__init__()

        self.criterion = criterion
        self.deep_base = deep_base

        resnet = resnet50(pretrained=pretrained, deep_base=True)
        self.resnet = resnet
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.conv2,
            resnet.bn2,
            resnet.relu,
            resnet.conv3,
            resnet.bn3,
            resnet.relu,
            resnet.maxpool,
        )

        self.cls = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # x: (N, C, H, W)
        _, _, H, W = x.shape

        # backbone
        x = self.layer0(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # classifier head
        x = self.cls(x)

        # upsample to input spatial size
        logits = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        # per-pixel prediction
        yhat = logits.argmax(dim=1)

        # IMPORTANT: only compute loss if y is provided
        if y is not None:
            main_loss = self.criterion(logits, y)
            # dummy aux_loss to match PSPNet-style API
            aux_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)
        else:
            main_loss = None
            aux_loss = None

        return logits, yhat, main_loss, aux_loss
