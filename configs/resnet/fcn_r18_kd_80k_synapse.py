from mmengine.config import read_base
from seg.models.backbones import ResNetV1c_KD
with read_base():
    from .fcn_r50_d8_80k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        type=ResNetV1c_KD,
        depth=18,
        out_channels=[64, 128, 256, 512],
        init_cfg=None),
    decode_head=dict(
        in_channels=512,
        channels=128)))
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='fcn-r18-kd-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
