from mmengine.config import read_base
from mmcv.cnn.bricks import ContextBlock


with read_base():
    from ..fcn.fcn_r34_d8_40k_flare22 import * # noqa

model.update(dict(
    backbone=dict(
        # init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='work_dirs/resnet50_gcb_in1k/best_accuracy/top1_76-15_epoch_93.pth'),
        plugins=[dict(
            cfg=dict(type=ContextBlock, ratio=1. / 4),
            stages=(False, True, True, True),
            position='after_conv2')]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='fcn-r34-gcb-40k'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

# optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
