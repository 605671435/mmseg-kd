from mmengine.config import read_base

with read_base():
    from .._base_.models.fcn_r50_d8 import *  # noqa
    from .._base_.datasets.cranium_512x512 import *  # noqa
    from .._base_.schedules.schedule_20k import *  # noqa
    from .._base_.default_runtime import *  # noqa
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=2,
        loss_decode=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True)),
    auxiliary_head=None)
)

train_dataloader.update(dict(batch_size=2, num_workers=2))
val_dataloader.update(dict(batch_size=1, num_workers=4))
test_dataloader = val_dataloader

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='cranium', name='fcn-r50-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
