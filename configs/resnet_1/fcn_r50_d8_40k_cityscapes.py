from mmengine.config import read_base

with read_base():
    from .._base_.models.fcn_r50_cityscapes import *  # noqa
    from .._base_.datasets.cityscapes import *  # noqa
    from .._base_.schedules.schedule_40k_cityscapes import *  # noqa
    from .._base_.default_runtime import *  # noqa
crop_size = (512, 1024)
data_preprocessor.update(dict(size=crop_size))
model.update(dict(data_preprocessor=data_preprocessor))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='cityscapes', name='fcn-r50-40k'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
