from mmengine.config import read_base
from seg.models.medical_seg.mednext.MedNextV1 import MedNeXt
from torch.optim import AdamW
from mmengine.optim.scheduler import LinearLR, CosineAnnealingLR
from seg.models.decode_heads.decode_head import LossHead
with read_base():
    from .._base_.models.unet_r18_d16_ds import *  # noqa
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(
    dict(data_preprocessor=data_preprocessor,
         pretrained=None,
         neck=None,
         decode_head=dict(num_classes=9),
         auxiliary_head=None,
         test_cfg=dict(mode='whole')))
model['backbone'] = dict(
    type=MedNeXt,
    in_channels=1,
    n_channels=32,
    n_classes=9,
    exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
    kernel_size=3,
    deep_supervision=False,
    do_res=True,
    do_res_up_down=True,
    block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
    dim='2d')
model['decode_head'] = dict(
    type=LossHead,
    num_classes=9,
    loss_decode=[
        dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
        dict(
            type=MemoryEfficientSoftDiceLoss, loss_weight=1.0)])
# optimizer
# optim_wrapper = dict(
#     type=OptimWrapper,
#     optimizer=dict(
#         type=AdamW, lr=4e-4, weight_decay=1e-5))
#
# param_scheduler = [
#     # 在 [0, 100) 迭代时使用线性学习率
#     dict(type=LinearLR,
#          start_factor=1e-6,
#          by_epoch=False,
#          begin=0,
#          end=500),
#     # 在 [100, 900) 迭代时使用余弦学习率
#     dict(type=CosineAnnealingLR,
#          T_max=39500,
#          by_epoch=False,
#          begin=500,
#          end=40000)
# ]

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='swin-unetr-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
