from mmengine.config import read_base
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from seg.models.segmentors.monai_model import MonaiModel

with read_base():
    from .._base_.datasets.synapse_3d import *  # noqa
    from .._base_.schedules.schedule_5000e_adamw import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# model settings
model = dict(
    type=MonaiModel,
    num_classes=14,
    model_cfg=dict(
        type=SwinUNETR,
        img_size=roi,
        feature_size=48,
        in_channels=1,
        out_channels=14,
        spatial_dims=3,
        use_checkpoint=True),
    loss_function=dict(
        type=DiceCELoss, to_onehot_y=True, softmax=True),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=4,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))

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
