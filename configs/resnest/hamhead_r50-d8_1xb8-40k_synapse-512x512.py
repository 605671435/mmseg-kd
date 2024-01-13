_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/synapse.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    decode_head=dict(
        _delete_=True,
        type='LightHamHead',
        in_channels=(512, 1024, 2048),
        in_index=(1, 2, 3),
        channels=512,
        ham_channels=512,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.2),
            dict(
                type='DiceLoss', loss_weight=0.8)],
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
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
