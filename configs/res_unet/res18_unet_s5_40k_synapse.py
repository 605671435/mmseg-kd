from mmengine.config import read_base
with read_base():
    from .._base_.models.deepsupervision_unet_r50_s7_d64 import *  # noqa
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        n_stages=5,
        features_per_stage=(32, 64, 128, 256, 512),
        conv_op=Conv2d,
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 2, 2, 2, 2),
        n_conv_per_stage_decoder=(1, 1, 1, 1)),
    auxiliary_head=None))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='res18-unet-s5-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
