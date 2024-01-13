from mmengine.config import read_base
with read_base():
    from .._base_.models.deepsupervision_unet_r50_s7_d64 import *  # noqa
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_80k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(dict(data_preprocessor=data_preprocessor))

default_hooks.update(dict(
    checkpoint=dict(
        type=MyCheckpointHook,
        by_epoch=False,
        interval=8000,
        max_keep_ckpts=1,
        save_best=['mDice'], rule='greater')))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='r50-unet-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
