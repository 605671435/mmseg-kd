from mmengine.config import read_base
with read_base():
    from .._base_.models.unet_r18_s4_d8 import *  # noqa
    from .._base_.datasets.FLARE22 import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(
    dict(data_preprocessor=data_preprocessor,
         pretrained=None,
         decode_head=dict(num_classes=14),
         auxiliary_head=None,
         test_cfg=dict(mode='whole')))

optim_wrapper.update(
    dict(
        optimizer=dict(
            lr=0.05)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='r18-unet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
