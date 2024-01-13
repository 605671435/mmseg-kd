from mmengine.config import read_base
from seg.models.utils import eca_layer


with read_base():
    from ..fcn.fcn_r18_d8_40k_lits17 import * # noqa

model.update(dict(
    backbone=dict(
        # init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='work_dirs/resnet50_ecanet_in1k/best_accuracy/top1_77-16_epoch_96.pth'),
        plugins=[dict(
                        cfg=dict(type=eca_layer),
                        stages=(True, True, True, True),
                        position='after_conv2')
                    ]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='lits17', name='fcn-r18-ecanet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

