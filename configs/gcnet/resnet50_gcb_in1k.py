from mmengine.config import read_base
from mmcv.cnn.bricks import ContextBlock
from seg.models.backbones import ResNet as plugin_resnet
with read_base():
    from ..resnet.resnet50_in1k import * # noqa

model.update(
    dict(
        backbone=dict(
            type=plugin_resnet,
            plugins=[dict(
                cfg=dict(type=ContextBlock, ratio=1. / 4),
                stages=(False, True, True, True),
                position='after_conv3')
            ])))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='imagenet1k', name='r50-gcb'))
]
log_processor = dict(by_epoch=True)
visualizer = dict(type=UniversalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
