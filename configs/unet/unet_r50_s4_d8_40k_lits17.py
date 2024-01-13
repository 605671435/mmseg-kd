from mmengine.config import read_base
with read_base():
    from .._base_.models.unet_r50_s4_d8 import *  # noqa
    from .._base_.datasets.lits17 import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
crop_size = (256, 256)
data_preprocessor.update(dict(size=crop_size))
model.update(
    dict(
        data_preprocessor=data_preprocessor,
        decode_head=dict(num_classes=3)))

default_hooks.update(dict(
    checkpoint=dict(
        type=MyCheckpointHook,
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=1,
        save_best=['mDice'], rule='greater')))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='lits17', name='r50-unet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
