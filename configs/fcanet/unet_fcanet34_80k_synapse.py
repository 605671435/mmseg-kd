from mmengine.config import read_base
from seg.models.backbones.fcanet_resnet import ResNetV1c as FcaNet_resnet

with read_base():
    from ..unet.unet_r34_s4_d8_40k_synapse import * # noqa

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
            project='synapse', name='unet-r34-fcanet'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

