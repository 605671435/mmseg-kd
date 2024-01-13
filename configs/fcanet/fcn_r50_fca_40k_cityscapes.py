from mmengine.config import read_base
from seg.models.backbones.fcanet_resnet import ResNetV1c as FcaNet_resnet

with read_base():
    from ..resnet.fcn_r50_d8_40k_cityscapes import * # noqa

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
            project='cityscapes', name='fcn-r50-fca'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

