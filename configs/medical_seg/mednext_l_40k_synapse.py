from mmengine.config import read_base

with read_base():
    from .mednext_40k_synapse import *  # noqa

# model settings
model['backbone'].update(
    dict(
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        kernel_size=5,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style='outside_block'))
# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=4e-4, weight_decay=1e-5))

param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=40000,
        eta_min=0.0,
        by_epoch=False,
    )
]

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='mednext-large-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
