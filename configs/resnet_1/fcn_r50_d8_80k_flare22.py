from mmengine.config import read_base

with read_base():
    from .._base_.models.fcn_r50_d8 import *  # noqa
    from .._base_.datasets.FLARE22 import *  # noqa
    from .._base_.schedules.schedule_80k import *  # noqa
    from .._base_.default_runtime import *  # noqa
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(
    dict(data_preprocessor=data_preprocessor,
         backbone=dict(init_cfg=None),
         decode_head=dict(num_classes=14),
         auxiliary_head=None))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='fcn-r50-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
