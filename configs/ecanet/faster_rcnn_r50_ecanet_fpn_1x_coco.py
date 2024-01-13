from mmengine.config import read_base
from seg.models.utils import eca_layer
# from seg.models.backbones import ResNet as plugin_resnet
with read_base():
    from ..resnet.faster_rcnn_r50_fpn_1x_coco import *  # noqa

model.update(
    dict(
        backbone=dict(
            # type=plugin_resnet,
            frozen_stages=1,
            init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='work_dirs/resnet50_ecanet_in1k/best_accuracy/top1_77-16_epoch_96.pth'),
            plugins=[dict(
                            cfg=dict(type=eca_layer),
                            stages=(True, True, True, True),
                            position='after_conv3')
                        ]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='coco', name='r50-ecanet'))
]
log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)
visualizer = dict(type=DetLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
