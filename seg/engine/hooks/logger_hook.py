# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from mmengine.fileio import FileClient, dump
from mmengine.fileio.io import get_file_backend
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS
from mmengine.utils import is_seq_of, scandir
from mmengine.hooks import LoggerHook
DATA_BATCH = Optional[Union[dict, tuple, list]]
SUFFIX_TYPE = Union[Sequence[str], str]


@HOOKS.register_module()
class MyLoggerHook(LoggerHook):
    def __init__(self, val_interval, **kwargs):
        super(MyLoggerHook, self).__init__(**kwargs)
        self.val_interval = val_interval

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Record logs after training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        # Print experiment name every n iterations.
        if self.every_n_train_iters(
                runner, self.interval_exp_name):
            exp_info = f'Exp name: {runner.experiment_name}'
            runner.logger.info(exp_info)
        if self.every_n_inner_iters(batch_idx, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
              and (not self.ignore_last
                   or len(runner.train_dataloader) <= self.interval)):
            # `runner.max_iters` may not be divisible by `self.interval`. if
            # `self.ignore_last==True`, the log of remaining iterations will
            # be recorded (Epoch [4][1000/1007], the logs of 998-1007
            # iterations will be recorded).
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        else:
            return
        runner.logger.info(log_str)
        runner.visualizer.add_scalars(
            tag, step=runner.iter + 1, file_path=self.json_log_path)

