from mmengine.config import read_base
from mmpretrain.models.backbones import ResNet_CIFAR
from mmpretrain.visualization import UniversalVisualizer
with read_base():
    from .._base_.models.resnet50 import *  # noqa
    from .._base_.datasets.cifar100 import *  # noqa
    from .._base_.schedules.cifar10_bs128 import *  # noqa
    from .._base_.default_runtime import *  # noqa

model.update(
    dict(
        backbone=dict(type=ResNet_CIFAR),
        head=dict(num_classes=100, topk=None)))

optim_wrapper.update(dict(optimizer=dict(weight_decay=0.0005)))
param_scheduler = dict(
    type=MultiStepLR,
    by_epoch=True,
    milestones=[60, 120, 160],
    gamma=0.2,
)
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='cifar100', name='r50'))
]
log_processor = dict(by_epoch=True)
visualizer = dict(type=UniversalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
