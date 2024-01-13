_base_ = [
    '../_base_/datasets/synapse.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.2),
            dict(
                type='DiceLoss', loss_weight=0.8)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', interval=1),
    checkpoint=dict(
                    type='MyCheckpointHook',
                    by_epoch=False,
                    interval=4000,
                    max_keep_ckpts=1,
                    save_best=['mDice'], rule='greater'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='pspnet-r50-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')

randomness = dict(seed=50000000,
                  deterministic=False,
                  diff_rank_seed=False)
