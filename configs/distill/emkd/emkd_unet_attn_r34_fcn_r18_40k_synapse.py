from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.architectures.connectors import TorchFunctionalConnector
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, \
    MethodOutputsRecorder
from mmrazor.models.losses import KLDivergence
from razor.models.losses.emkd_losses import IMD, RAD
# _stack_batch_gt
with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_40k import *  # noqa
    from ..._base_.default_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unet_attn_r34_s4_d8_40k_synapse import model as teacher_model  # noqa
    from ...resnet.fcn_r18_no_pretrain_80k_synapse import model as student_model  # noqa

teacher_ckpt = 'ckpts/unet_attn_r34_s4_d8_40k_synapse/best_mDice_88-31_iter_40000.pth'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            low_feat=dict(type=ModuleOutputsRecorder, source='backbone.stem'),
            high_feat=dict(type=ModuleOutputsRecorder, source='backbone.layer2'),
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg'),
            gt=dict(type=MethodOutputsRecorder,
                    source='seg.models.decode_heads.decode_head.BaseDecodeHead._stack_batch_gt')),
        teacher_recorders=dict(
            low_feat=dict(type=ModuleOutputsRecorder, source='backbone.stem'),
            high_feat=dict(type=ModuleOutputsRecorder, source='backbone.layer2'),
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        connectors=dict(
            preds_S=dict(type=TorchFunctionalConnector,
                         function_name='interpolate',
                         func_args=dict(size=128))),
        distill_losses=dict(
            loss_kl=dict(type=KLDivergence, tau=4, loss_weight=0.1, reduction='mean'),
            loss_imd_low=dict(type=IMD, loss_weight=0.9),
            loss_imd_high=dict(type=IMD, loss_weight=0.9),
            loss_rad_low=dict(type=RAD, num_classes=9, loss_weight=0.9),
            loss_rad_high=dict(type=RAD, num_classes=9, loss_weight=0.9),
        ),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='logits', connector='preds_S'),
                preds_T=dict(from_student=False, recorder='logits')),
            loss_imd_low=dict(
                logits_S=dict(from_student=True, recorder='low_feat'),
                logits_T=dict(from_student=False, recorder='low_feat')),
            loss_imd_high=dict(
                logits_S=dict(from_student=True, recorder='high_feat'),
                logits_T=dict(from_student=False, recorder='high_feat')),
            loss_rad_low=dict(
                logits_S=dict(from_student=True, recorder='low_feat'),
                logits_T=dict(from_student=False, recorder='low_feat'),
                ground_truth=dict(from_student=True, recorder='gt')),
            loss_rad_high=dict(
                logits_S=dict(from_student=True, recorder='high_feat'),
                logits_T=dict(from_student=False, recorder='high_feat'),
                ground_truth=dict(from_student=True, recorder='gt'))
            )))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='emkd_unet_attn_r34_fcn_r18-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer.update(
    dict(type=SegLocalVisualizer,
         vis_backends=vis_backends,
         name='visualizer'))
