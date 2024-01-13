from mmengine.config import read_base
with read_base():
    from .._base_.models.deepsupervision_unet_r18_s4_d8 import *  # noqa
    from .._base_.datasets.FLARE22 import *  # noqa
    from .._base_.schedules.schedule_160k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(dict(data_preprocessor=data_preprocessor))

model['decode_head'].update(dict(num_classes=14))
model['auxiliary_head'][0].update(dict(num_classes=14))
model['auxiliary_head'][1].update(dict(num_classes=14))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='unet-r18-ds-160k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
