from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from mmrazor.models.losses import KLDivergence
from seg.engine.hooks import MyCheckpointHook

with read_base():
    from ..._base_.datasets.lits17 import *  # noqa
    from ..._base_.schedules.schedule_40k import *  # noqa
    from ..._base_.default_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unet_r50_s4_d8_40k_lits17 import model as teacher_model  # noqa
    from ...unet.unet_r18_s4_d8_40k_lits17 import model as student_model  # noqa

teacher_ckpt = 'ckpts/unet_r50_s4_d8_40k_lits17/best_mDice_87-27_iter_40000.pth'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        distill_losses=dict(
            loss_kl=dict(type=KLDivergence, tau=2, loss_weight=10.0, reduction='mean')),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='lits17', name='kd_unet_r50_unet_r18-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer.update(
    dict(type=SegLocalVisualizer,
         vis_backends=vis_backends,
         name='visualizer'))
