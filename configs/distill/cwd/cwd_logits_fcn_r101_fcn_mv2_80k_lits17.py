from mmengine.config import read_base

with read_base():
    from .cwd_logits_fcn_r101_fcn_r18_80k_lits17 import *  # noqa
    from ...mobilenet_v2.mobilenet_v2_fcn_80k_lits17 import model as student_model  # noqa

model['architecture'] = dict(cfg_path=student_model)

# model.update(dict(
#     architecture=dict(cfg_path=student_model)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='lits17', name='cwd_fcn_r101_fcn_mv2-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
