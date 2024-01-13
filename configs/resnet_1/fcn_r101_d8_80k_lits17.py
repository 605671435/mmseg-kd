from mmengine.config import read_base
with read_base():
    from .fcn_r50_d8_80k_lits17 import * # noqa

model.update(dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://resnet101_v1c')),
    auxiliary_head=None))
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='lits17', name='fcn-r101-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
