_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/brats19.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(
    _delete_=True,
    type='SegDataPreProcessor',
    size=(128, 128, 128),
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(in_channels=4,
                  conv_cfg=dict(type='Conv3d'),
                  norm_cfg=dict(type='BN3d', requires_grad=True)),
    decode_head=dict(num_classes=4,
                     conv_cfg=dict(type='Conv3d'),
                     norm_cfg=dict(type='BN3d', requires_grad=True),
                     resize_mode='trilinear',
                     loss_decode=dict(
                         type='DiceLoss', ignore_index=3, loss_weight=1.0)),
    auxiliary_head=None,
    test_cfg=dict(mode='whole3d')
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False,
    )
]
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(
                    type='MyCheckpointHook',
                    by_epoch=False,
                    interval=4000,
                    max_keep_ckpts=1,
                    save_best=['mDice'], rule='greater'))

custom_hooks = [dict(type='DeterministicHook',
                     deterministic=False,
                     warn_only=True,
                     cublas_cfg=':4096:8')]

vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(
    #     type='WandbVisBackend',
    #     init_kwargs=dict(
    #         project='synapse', name='fcn-r50-40k'),
    #     define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')

randomness = dict(seed=50000000,
                  deterministic=False,
                  diff_rank_seed=False)
