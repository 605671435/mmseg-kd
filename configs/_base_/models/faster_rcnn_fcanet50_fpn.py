from mmdet.models.detectors import FasterRCNN
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.models.necks import FPN
from mmdet.models.dense_heads import RPNHead
from mmdet.models.task_modules.prior_generators import AnchorGenerator
from mmdet.models.task_modules.coders import DeltaXYWHBBoxCoder
from mmdet.models.roi_heads import Shared2FCBBoxHead, StandardRoIHead, \
    SingleRoIExtractor
from mmdet.models.losses import CrossEntropyLoss, L1Loss
from mmdet.models.task_modules.assigners import MaxIoUAssigner
from mmdet.models.task_modules.samplers import RandomSampler
from mmcv.ops import RoIAlign, nms
from pretrain.models.backbones.freqnet import FreqNet
# model settings
model = dict(
    type=FasterRCNN,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type=FreqNet,
        depth=50,
        out_indices=(0, 1, 2, 3)),
    neck=dict(
        type=FPN,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type=RPNHead,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type=AnchorGenerator,
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type=DeltaXYWHBBoxCoder,
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type=CrossEntropyLoss, use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type=L1Loss, loss_weight=1.0)),
    roi_head=dict(
        type=StandardRoIHead,
        bbox_roi_extractor=dict(
            type=SingleRoIExtractor,
            roi_layer=dict(type=RoIAlign, output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type=Shared2FCBBoxHead,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type=DeltaXYWHBBoxCoder,
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type=MaxIoUAssigner,
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type=RandomSampler,
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type=nms, iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type=MaxIoUAssigner,
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type=RandomSampler,
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type=nms, iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type=nms, iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
