_base_ = ['../resnet/fcn_r50-d8_1xb8-40k_synapse-512x512.py']

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='ResNeSt',
        depth=101,
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://resnest101')),
    decode_head=dict(num_classes=9),
    # auxiliary_head=dict(num_classes=14),
    auxiliary_head=None,
    test_cfg=dict(mode='whole')
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False,
    )
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
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
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='fcn-r50-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')

randomness = dict(seed=50000000,
                  deterministic=False,
                  diff_rank_seed=False)
