# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
from functools import partial
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist, track_parallel_progress, track_progress
from PIL import Image
from prettytable import PrettyTable
from .confusion_matrix import *
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch


class HD95Metric(HausdorffDistanceMetric):

    def __init__(self, **kwargs):
        super().__init__(percentile=95, **kwargs)


mapping = dict(
    Dice=DiceMetric,
    IoU=MeanIoU,
    HD95=HD95Metric)


class MonaiMetric(BaseMetric):
    def __init__(self,
                 metrics: List[str],
                 num_classes: int,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 include_background: bool = True,
                 get_not_nans: bool = True,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.post_label = AsDiscrete(to_onehot=self.num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        self.metric_names = metrics
        self.metrics = []
        for metric in metrics:
            metric = mapping[metric](
                include_background=include_background,
                get_not_nans=get_not_nans)
            self.metrics.append(metric)

    def process(self, data_batch: torch.Tensor, data_samples: dict) -> None:
        target = data_samples['label'].cuda()
        if not data_batch.is_cuda:
            target = target.cpu()
        val_labels_list = decollate_batch(target)
        val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        val_outputs_list = decollate_batch(data_batch)
        val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        results_list = dict()
        for name, metric_func in zip(self.metric_names, self.metrics):
            metric_func.reset()
            metric_func(y_pred=val_output_convert, y=val_labels_convert)
            # results, not_nans = metric_func.aggregate()
            results = metric_func.get_buffer()
            results_list[name] = results[0].detach().cpu().numpy()
        ret_metrics = self.format_metrics(results_list)
        self.results.append(ret_metrics)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        metrics = dict()
        for key in results[0].keys():
            metrics[key] = np.round(np.nanmean([case_metric[key] for case_metric in results]), 2)

        return metrics

    def format_metrics(self, ret_metrics: dict) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        class_names = self.dataset_meta['classes']

        ret_metrics_summary = OrderedDict({
            name: np.nanmean(ret_metrics[name])
            for name in self.metric_names})

        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'HD95':
                ret_metrics_summary[key] = np.round(val * 1, 2)
            else:
                ret_metrics_summary[key] = np.round(val * 100, 2)
            metrics[key] = ret_metrics_summary[key]

        # each class table
        ret_metrics_class = OrderedDict()
        for name in self.metric_names:
            if name == 'HD95':
                ret_metrics_class[name] = np.round(ret_metrics[name] * 1, 2)
            else:
                ret_metrics_class[name] = np.round(ret_metrics[name] * 100, 2)
            ret_metrics_class[name] = np.round(np.append(ret_metrics_class[name], ret_metrics_summary[name]), 2)
        for name in self.metric_names:  # ['Dice', 'Jaccard', 'HD95']
            for class_key, metric_value in zip(class_names, ret_metrics_class[name]):
                metrics[f'{name} ({class_key})'] = metric_value

        ret_metrics_class.update({'Class': class_names + ('Average',)})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        return metrics
