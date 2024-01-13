_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/lits17.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    decode_head=dict(num_classes=3),
    auxiliary_head=None,
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(170, 170))
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

train_dataloader = dict(batch_size=8, num_workers=4)
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
            project='lits17', name='fcn-r50-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')

randomness = dict(seed=50000000,
                  deterministic=False,
                  diff_rank_seed=False)
