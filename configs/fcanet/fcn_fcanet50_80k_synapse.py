from mmengine.config import read_base
from seg.models.backbones.fcanet_resnet import ResNetV1c as FcaNet_resnet

with read_base():
    from ..fcn.fcn_r50_d8_80k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        type=FcaNet_resnet,
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='fcn-fcanet50-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

