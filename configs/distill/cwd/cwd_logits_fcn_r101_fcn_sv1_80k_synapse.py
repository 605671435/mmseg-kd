from mmengine.config import read_base

with read_base():
    from .cwd_logits_fcn_r101_fcn_r18_80k_synapse import *  # noqa
    from ...shufflenet_v1.shufflenet_v1_fcn_80k_synapse import model as student_model  # noqa

model.update(dict(
    architecture=dict(cfg_path=student_model)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='exkd_fcn_r101_fcn_sv1-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
