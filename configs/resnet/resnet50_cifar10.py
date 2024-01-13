from mmengine.config import read_base
from mmpretrain.models.backbones import ResNet_CIFAR
from mmpretrain.visualization import UniversalVisualizer
with read_base():
    from .._base_.models.resnet50 import *  # noqa
    from .._base_.datasets.cifar10 import *  # noqa
    from .._base_.schedules.cifar10_bs128 import *  # noqa
    from .._base_.default_runtime import *  # noqa

model.update(
    dict(
        backbone=dict(type=ResNet_CIFAR),
        head=dict(num_classes=10, topk=None)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='cifar10', name='r50'))
]
log_processor = dict(by_epoch=True)
visualizer = dict(type=UniversalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
