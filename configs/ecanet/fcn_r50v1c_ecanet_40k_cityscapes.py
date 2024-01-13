from mmengine.config import read_base
from seg.models.utils import eca_layer


with read_base():
    from ..resnet.fcn_r50_d8_40k_cityscapes import * # noqa

model.update(dict(
    backbone=dict(
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
            project='cityscapes', name='fcn-r50v1c-ecanet-40k'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

# optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
