from mmengine.config import read_base
with read_base():
    from .unet_r18_d16_40k_synapse import *  # noqa
# model settings
model.update(
    dict(backbone=dict(
        depth=50,
        init_cfg=None
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
        #     prefix='backbone')
    )))
model['neck'].update(dict(base_channels=256))
model['decode_head'].update(
    dict(in_channels=256,
         channels=256))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-r50-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
