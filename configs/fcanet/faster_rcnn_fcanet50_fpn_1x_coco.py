from mmengine.config import read_base
from mmengine.runner import LogProcessor
from mmdet.visualization import DetLocalVisualizer
with read_base():
    from .._base_.models.faster_rcnn_fcanet50_fpn import *  # noqa
    from .._base_.datasets.coco_detection import *  # noqa
    from .._base_.schedules.schedule_1x import *  # noqa
    from .._base_.default_runtime import *  # noqa

auto_scale_lr = dict(enable=True, base_batch_size=16)

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='coco', name='fcanet50'))
]
log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)
visualizer = dict(type=DetLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
