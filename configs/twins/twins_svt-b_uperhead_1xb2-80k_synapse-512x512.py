_base_ = [
    '../_base_/datasets/synapse.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/alt_gvt_base_20220308-1b7eb711.pth'   # noqa

# model settings
backbone_norm_cfg = dict(type='LN')
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=None,
    std=None,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SVT',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=3,
        embed_dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        patch_sizes=[4, 2, 2, 2],
        strides=[4, 2, 2, 2],
        mlp_ratios=[4, 4, 4, 4],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        norm_cfg=backbone_norm_cfg,
        depths=[2, 2, 18, 2],
        windiow_sizes=[7, 7, 7, 7],
        sr_ratios=[8, 4, 2, 1],
        norm_after_stage=True,
        drop_rate=0.0,
        attn_drop_rate=0.,
        drop_path_rate=0.2),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.)
    }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=2, num_workers=2)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='twins_svt_b-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')

default_hooks = dict(
    checkpoint=dict(type='MyCheckpointHook',
                    by_epoch=False,
                    interval=8000,
                    max_keep_ckpts=1,
                    save_best=['mDice'], rule='greater'))