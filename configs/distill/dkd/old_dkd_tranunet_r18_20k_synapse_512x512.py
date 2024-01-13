from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder,ModuleInputsRecorder
from mmrazor.models.architectures.connectors import ConvModuleConnector
from razor.models.losses.decoupled_kd import DKDLoss
with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_20k import *  # noqa
    from ..._base_.default_runtime import *  # noqa

teacher_ckpt = "/home/jz207/workspace/zhangdw/decoupled_self_attention/vis_ckpts/transunet_40k_synapse/best_mDice_82-92_iter_40000.pth"
teacher_cfg_path = "/home/jz207/workspace/zhangdw/decoupled_self_attention/new_configs/medical_seg/transunet_40k_synapse.py"  # noqa: E501
student_cfg_path = '/home/jz207/workspace/zhangdw/decoupled_self_attention/new_configs/resnet/fcn_r18_d8_40k_synapse.py'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    # data_preprocessor=dict(
    #     type='ImgDataPreprocessor',
    #     # RGB format normalization parameters
    #     mean=[123.675, 116.28, 103.53],
    #     std=[58.395, 57.12, 57.375],
    #     # convert image from BGR to RGB
    #     bgr_to_rgb=True),
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg'),
            gt_labels=dict(type=ModuleInputsRecorder, source='decode_head.loss_decode.0')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='backbone.segmentation_head')),
        distill_losses=dict(
            loss_dkd=dict(
                type=DKDLoss,
                tau=4,
                alpha=1,
                beta=20,
                loss_weight=1e-5,
                reduction='mean')),
        loss_forward_mappings=dict(
            loss_dkd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
                gt_labels=dict(
                    recorder='gt_labels', from_student=True, data_idx=1)))))


find_unused_parameters = True

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='kd_fcn_r50_fcn_r18-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')

default_hooks = dict(
    checkpoint=dict(type='MyCheckpointHook',
                    by_epoch=False,
                    interval=2000,
                    max_keep_ckpts=1,
                    save_best=['mDice'], rule='greater'))

work_dir='/home/jz207/workspace/zhangdw/decoupled_self_attention/work_dirs_transunt/old_dkd_tranunet_r18_20k_synapse-512x512/teacher_en_de_new_20k_tau4_alpha1_beta20_lw1e_5'

