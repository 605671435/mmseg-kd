from mmengine.config import read_base
with read_base():
    from .fcn_r50_d8_40k_flare22 import * # noqa

model.update(dict(
    backbone=dict(
        depth=34),
    decode_head=dict(
        in_channels=512,
        channels=128)))
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='fcn-r34-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer.update(
    dict(vis_backends=vis_backends,
         name='visualizer'))
