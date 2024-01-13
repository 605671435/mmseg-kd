from mmengine.config import read_base
from mmpretrain.visualization import UniversalVisualizer
with read_base():
    from .._base_.models.resnet50 import *  # noqa
    from .._base_.datasets.imagenet_bs32 import *  # noqa
    from .._base_.schedules.imagenet_bs256 import *  # noqa
    from .._base_.default_runtime import *  # noqa

model.update(
    dict(
        backbone=dict(depth=18),
        head=dict(in_channels=512)))
    
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='imagenet1k', name='r18'))
]
log_processor = dict(by_epoch=True)
visualizer = dict(type=UniversalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
