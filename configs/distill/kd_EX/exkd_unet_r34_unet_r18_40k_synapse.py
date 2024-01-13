from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import EXDistiller
from razor.models.architectures.connectors import EXKDConnector
from mmrazor.models.architectures.connectors import TorchFunctionalConnector
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from mmrazor.models.losses import L2Loss, KLDivergence

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_40k import *  # noqa
    from ..._base_.default_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unet_r34_s4_d8_40k_synapse import model as teacher_model  # noqa
    from ...resnet.fcn_r18_kd_80k_synapse import model as student_model  # noqa

teacher_ckpt = 'ckpts/unet_attn_r34_s4_d8_40k_synapse/best_mDice_88-31_iter_40000.pth'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=EXDistiller,
        student_recorders=dict(
            bb_l1_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer1.2.conv1'),
            bb_l2_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer2.2.conv1'),
            bb_l3_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.2.conv1'),
            bb_l4_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer4.2.conv1'),
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        teacher_recorders=dict(
            bb_l1_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer1.2.conv2'),

            bb_l2_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer2.2.conv2'),
            bb_l2_3_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer2.3.conv2'),

            bb_l3_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.2.conv2'),
            bb_l3_3_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.3.conv2'),
            bb_l3_4_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.4.conv2'),
            bb_l3_5_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer3.5.conv2'),

            bb_l4_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer4.2.conv2'),
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        connectors=dict(
            preds_S=dict(type=TorchFunctionalConnector,
                         function_name='interpolate',
                         func_args=dict(size=128))),
        distill_losses=dict(
            loss_1=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_2=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_3=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_4=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_kl=dict(type=KLDivergence, tau=2, loss_weight=0.1, reduction='mean')),
        # connectors=dict(
        #     loss_s1_sfeat=dict(type=EXKDConnector,
        #                        student_channels=64,
        #                        teacher_channels=256),
        #     loss_s2_sfeat=dict(type=EXKDConnector,
        #                        student_channels=128,
        #                        teacher_channels=512),
        #     loss_s3_sfeat=dict(type=EXKDConnector,
        #                        student_channels=256,
        #                        teacher_channels=1024),
        #     loss_s4_sfeat=dict(type=EXKDConnector,
        #                        student_channels=512,
        #                        teacher_channels=2048)),
        loss_forward_mappings=dict(
            loss_1=dict(
                s_feature=dict(
                    recorder='bb_l1_1_bn3',
                    from_student=True),
                t_feature=dict(recorder='bb_l1_2_bn3', from_student=False)),
            loss_2=dict(
                s_feature=dict(
                    recorder='bb_l2_1_bn3',
                    from_student=True),
                t_feature=dict(recorder=['bb_l2_2_bn3',
                                         'bb_l2_3_bn3'],
                               from_student=False)),
            loss_3=dict(
                s_feature=dict(
                    recorder='bb_l3_1_bn3',
                    from_student=True),
                t_feature=dict(recorder=['bb_l3_2_bn3',
                                         'bb_l3_3_bn3',
                                         'bb_l3_4_bn3',
                                         'bb_l3_5_bn3'],
                               from_student=False)),
            loss_4=dict(
                s_feature=dict(
                    recorder='bb_l4_1_bn3',
                    from_student=True),
                t_feature=dict(recorder='bb_l4_2_bn3', from_student=False)),
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='logits', connector='preds_S'),
                preds_T=dict(from_student=False, recorder='logits')))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='exkd_unet_attn_r34_fcn_r18-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
