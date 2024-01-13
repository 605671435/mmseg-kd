from mmengine.config import read_base
with read_base():
    from .fcn_r50_d8_20k_cranium import * # noqa

model.update(dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(
            type=PretrainedInit,
            checkpoint='open-mmlab://resnet18_v1c')),
    decode_head=dict(
        in_channels=512,
        channels=128)))
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='cranium', name='fcn-r18-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer.update(
    dict(vis_backends=vis_backends,
         name='visualizer'))
