# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn

from mmseg.models.utils import resize
from mmrazor.registry import MODELS
from mmrazor.models.architectures.connectors.base_connector import BaseConnector


@MODELS.register_module()
class EXKDConnector(BaseConnector):

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        student_shape: Optional[int] = None,
        teacher_shape: Optional[int] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        else:
            self.align = None
        self.student_shape = student_shape
        self.teacher_shape = teacher_shape

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        if self.student_shape is not None and self.teacher_shape is not None:
            feature = resize(feature,
                             size=self.teacher_shape,
                             mode='nearest')
        if self.align is not None:
            feature = self.align(feature)

        return feature

@MODELS.register_module()
class AddConnector(BaseConnector):

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        if self.align is not None:
            feature = self.align(feature)

        return feature