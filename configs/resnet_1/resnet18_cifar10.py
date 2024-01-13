from mmengine.config import read_base

with read_base():
    from .resnet50_cifar10 import *  # noqa

model.update(
    dict(
        backbone=dict(depth=18),
        head=dict(in_channels=512)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='cifar10', name='r18'))
]
log_processor = dict(by_epoch=True)
visualizer = dict(type=UniversalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
