from mmengine.config import read_base
from mmengine.runner import LogProcessor
from mmdet.visualization import DetLocalVisualizer
with read_base():
    from .._base_.models.faster_rcnn_fcanet50_fpn import *  # noqa
    from .._base_.datasets.coco_semantic import *  # noqa
    from .._base_.schedules.schedule_1x import *  # noqa
    from .._base_.default_runtime import *  # noqa

# optimizer
# lr is set for a batch size of 8
optim_wrapper.update(dict(optimizer=dict(lr=0.01)))

# learning rate
param_scheduler = [
    dict(
        type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=8,
        by_epoch=True,
        # [7] yields higher performance than [6]
        milestones=[7],
        gamma=0.1)
]

# actual epoch = 8 * 8 = 64
train_cfg.update(dict(max_epochs=8))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
# TODO: support auto scaling lr
auto_scale_lr = dict(base_batch_size=16)

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='coco_semantic', name='fcanet50'))
]
log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)
visualizer = dict(type=DetLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
