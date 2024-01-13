from mmengine.config import read_base
with read_base():
    from .unet_r18_d16_ds_40k_synapse import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(
    dict(
        auxiliary_head=None,
        decode_head=dict(
            loss_decode=[
                dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
                dict(
                    type=MemoryEfficientSoftDiceLoss, loss_weight=1.0)])))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-r18-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
