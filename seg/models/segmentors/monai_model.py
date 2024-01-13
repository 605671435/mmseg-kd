# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional
from urllib.parse import urlparse
import os
from functools import partial

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

from seg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from typing import Dict, Optional, Tuple, Union

from mmengine.optim import OptimWrapper
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction

class MonaiModel(nn.Module):

    def __init__(self,
                 model_cfg: ConfigType,
                 loss_function: ConfigType,
                 num_classes: int,
                 infer_cfg: Optional[ConfigType] = None):
        super().__init__()
        self.model = MODELS.build(model_cfg)
        if infer_cfg is not None:
            self.model_inferer = partial(
                sliding_window_inference,
                roi_size=infer_cfg.inf_size,
                sw_batch_size=infer_cfg.sw_batch_size,
                predictor=self.model,
                overlap=infer_cfg.infer_overlap)
        else:
            self.model_inferer = None
        self.loss_function = MODELS.build(loss_function)
        self.post_label = AsDiscrete(to_onehot=num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.acc_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)

    def train_step(self, batch_data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
        with optim_wrapper.optim_context(self):
            logit_map = self.model(data)
            losses = self.loss_function(logit_map, target)
        optim_wrapper.update_params(losses)
        return dict(loss=losses)

    def val_step(self, batch_data: Union[dict, tuple, list]) -> Tuple[np.ndarray]:

        data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
        if self.model_inferer is not None:
            logits = self.model_inferer(data)
        else:
            logits = self.model(data)
        if not logits.is_cuda:
            target = target.cpu()
        val_labels_list = decollate_batch(target)
        val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        self.acc_func.reset()
        self.acc_func(y_pred=val_output_convert, y=val_labels_convert)
        acc, not_nans = self.acc_func.aggregate()
        acc = acc.cuda()
        return acc.detach().cpu().numpy(), not_nans.detach().cpu().numpy()
        # return logits

    def test_step(self, batch_data: Union[dict, tuple, list]) -> torch.Tensor:

        data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
        if self.model_inferer is not None:
            logits = self.model_inferer(data)
        else:
            logits = self.model(data)
        # if not logits.is_cuda:
        #     target = target.cpu()
        # val_labels_list = decollate_batch(target)
        # val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        # val_outputs_list = decollate_batch(logits)
        # val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        # self.acc_func.reset()
        # self.acc_func(y_pred=val_output_convert, y=val_labels_convert)
        # acc, not_nans = self.acc_func.aggregate()
        # acc = acc.cuda()
        # return acc.detach().cpu().numpy(), not_nans.detach().cpu().numpy()
        return logits
