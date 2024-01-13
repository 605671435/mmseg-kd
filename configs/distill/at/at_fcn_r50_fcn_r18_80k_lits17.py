from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from mmrazor.models.losses import ATLoss
from seg.engine.hooks import MyCheckpointHook

with read_base():
    from ..._base_.datasets.lits17 import *  # noqa
    from ..._base_.schedules.schedule_80k import *  # noqa
    from ..._base_.default_runtime import *  # noqa
    # import teacher model and student model
    from ...resnet.fcn_r50_d8_80k_lits17 import model as teacher_model  # noqa
    from ...resnet.fcn_r18_no_pretrain_80k_lits17 import model as student_model  # noqa

teacher_ckpt = 'ckpts/fcn_r50-d8_1xb2-80k_lits17-256x256/best_mDice_80-63_iter_72000.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            bb_s1=dict(type=ModuleOutputsRecorder, source='backbone.layer1.1.relu'),
            bb_s2=dict(type=ModuleOutputsRecorder, source='backbone.layer2.1.relu'),
            bb_s3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.1.relu'),
            bb_s4=dict(type=ModuleOutputsRecorder, source='backbone.layer4.1.relu')),
        teacher_recorders=dict(
            bb_s1=dict(type=ModuleOutputsRecorder, source='backbone.layer1.2.relu'),
            bb_s2=dict(type=ModuleOutputsRecorder, source='backbone.layer2.3.relu'),
            bb_s3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.5.relu'),
            bb_s4=dict(type=ModuleOutputsRecorder, source='backbone.layer4.2.relu')),
        distill_losses=dict(
            loss_s1=dict(type=ATLoss, loss_weight=250.0),
            loss_s2=dict(type=ATLoss, loss_weight=250.0),
            loss_s3=dict(type=ATLoss, loss_weight=250.0),
            loss_s4=dict(type=ATLoss, loss_weight=250.0)),
        loss_forward_mappings=dict(
            loss_s1=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s1', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='bb_s1', record_idx=2)),
            loss_s2=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s2', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='bb_s2', record_idx=2)),
            loss_s3=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s3', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='bb_s3', record_idx=2)),
            loss_s4=dict(
                s_feature=dict(
                    from_student=True, recorder='bb_s4', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='bb_s4', record_idx=2)))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='lits17', name='at_fcn_r50_fcn_r18-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')

default_hooks.update(dict(
    checkpoint=dict(type=MyCheckpointHook,
                    by_epoch=False,
                    interval=8000,
                    max_keep_ckpts=1,
                    save_best=['mDice'], rule='greater')))
