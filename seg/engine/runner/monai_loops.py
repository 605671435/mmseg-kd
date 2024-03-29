# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader
from mmengine.evaluator import Evaluator
from seg.registry import LOOPS, EVALUATOR
from mmengine.runner.amp import autocast
from mmengine.runner import BaseLoop, ValLoop, TestLoop
from mmengine.registry import DATASETS
from mmengine.logging import MMLogger, print_log


@LOOPS.register_module()
class MonaiValLoop(ValLoop):

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        outputs = []
        for idx, data_batch in enumerate(self.dataloader):
            outputs.append(self.run_iter(idx, data_batch))

        # compute metrics
        mean_acc = np.round(np.nanmean([output[0] for output in outputs])*100, 2)
        metrics = dict(mDice=mean_acc)
        # metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        # self.evaluator.process(data_batch=outputs, data_samples=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        return outputs

@LOOPS.register_module()
class MonaiTestLoop(ValLoop):

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        outputs = []
        logger: MMLogger = MMLogger.get_current_instance()
        for idx, data_batch in enumerate(self.dataloader):
            print_log(f'Test {idx + 1}/{len(self.dataloader)}', logger)
            output = self.run_iter(idx, data_batch)
        # compute metrics
        # mean_acc = np.round(np.nanmean([output[0] for output in outputs])*100, 2)
        # metrics = dict(mDice=mean_acc)
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
        self.evaluator.process(data_batch=outputs, data_samples=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        return outputs