from mmengine.config import read_base
from seg.models.selfsup.selfsup_swin_unetr import SwinUNETR_SelfSupervisor
from seg.models.medical_seg.pretrain_swin_unetr.ssl_head import SSLHead
from torch.optim import AdamW
from mmengine.optim.scheduler import LinearLR
from seg.engine.runner.loops import SelfSupValLoop, SelfSupTestLoop

with read_base():
    from .._base_.models.unet_r18_s4_d8 import *  # noqa
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_80k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model = dict(
    type=SwinUNETR_SelfSupervisor,
    batch_size=1,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=SSLHead,
        in_channels=1))

train_dataloader.update(dict(batch_size=1))
# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=1e-4, weight_decay=1e-5))

param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]
train_cfg = dict(type=IterBasedTrainLoop, max_iters=80000, val_interval=8000, val_begin=80000)
val_cfg = dict(type=SelfSupValLoop)
test_cfg = dict(type=SelfSupTestLoop)

default_hooks.update(
    dict(
        checkpoint=dict(
            type=MyCheckpointHook,
            by_epoch=False,
            interval=8000,
            max_keep_ckpts=1,
            save_best=['total_loss'], rule='less')))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='swin-unetr-40k'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
