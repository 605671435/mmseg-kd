from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import EXDistiller
from razor.models.architectures.connectors import EXKDConnector
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from razor.models.losses import EXKD_Loss
from seg.engine.hooks import MyCheckpointHook

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_80k import *  # noqa
    from ..._base_.default_runtime import *  # noqa
    # import teacher model and student model
    from ...convnext.conv_next_b_synapse import model as teacher_model  # noqa
    from ...resnet.fcn_r18_kd_80k_synapse import model as student_model  # noqa

teacher_ckpt = 'ckpts/conv_next_b-synapse-80k/best_mDice_84-31_iter_72000.pth'  # noqa: E501
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
            bb_l4_1_bn3=dict(type=ModuleOutputsRecorder, source='backbone.layer4.2.conv1')),
        teacher_recorders=dict(
            bb_l1_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.0.2.drop_path'),

            bb_l2_3_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.1.2.drop_path'),

            bb_l3_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.2.drop_path'),
            bb_l3_3_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.3.drop_path'),
            bb_l3_4_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.4.drop_path'),
            bb_l3_5_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.5.drop_path'),
            bb_l3_6_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.6.drop_path'),
            bb_l3_7_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.7.drop_path'),
            bb_l3_8_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.8.drop_path'),
            bb_l3_9_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.9.drop_path'),
            bb_l3_10_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.10.drop_path'),
            bb_l3_11_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.11.drop_path'),
            bb_l3_12_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.12.drop_path'),
            bb_l3_13_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.13.drop_path'),
            bb_l3_14_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.14.drop_path'),
            bb_l3_15_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.15.drop_path'),
            bb_l3_16_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.16.drop_path'),
            bb_l3_17_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.17.drop_path'),
            bb_l3_18_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.18.drop_path'),
            bb_l3_19_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.19.drop_path'),
            bb_l3_20_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.20.drop_path'),
            bb_l3_21_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.21.drop_path'),
            bb_l3_22_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.22.drop_path'),
            bb_l3_23_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.23.drop_path'),
            bb_l3_24_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.24.drop_path'),
            bb_l3_25_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.25.drop_path'),
            bb_l3_26_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.26.drop_path'),

            bb_l4_2_bn3=dict(type=ModuleOutputsRecorder, source='backbone.stages.3.2.drop_path')),
        distill_losses=dict(
            loss_1=dict(type=EXKD_Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_2=dict(type=EXKD_Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_3=dict(type=EXKD_Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_4=dict(type=EXKD_Loss, normalize=True, loss_weight=0.5, dist=True)),
        connectors=dict(
            loss_s1_sfeat=dict(type=EXKDConnector,
                               student_channels=64,
                               teacher_channels=128),
            loss_s2_sfeat=dict(type=EXKDConnector,
                               student_channels=128,
                               teacher_channels=256),
            loss_s3_sfeat=dict(type=EXKDConnector,
                               student_channels=256,
                               teacher_channels=512,
                               student_shape=64,
                               teacher_shape=32),
            loss_s4_sfeat=dict(type=EXKDConnector,
                               student_channels=512,
                               teacher_channels=1024,
                               student_shape=64,
                               teacher_shape=16)),
        loss_forward_mappings=dict(
            loss_1=dict(
                s_feature=dict(
                    recorder='bb_l1_1_bn3',
                    from_student=True,
                    connector='loss_s1_sfeat'),
                t_feature=dict(recorder='bb_l1_2_bn3', from_student=False)),
            loss_2=dict(
                s_feature=dict(
                    recorder='bb_l2_1_bn3',
                    from_student=True,
                    connector='loss_s2_sfeat'),
                t_feature=dict(recorder='bb_l2_3_bn3', from_student=False)),
            loss_3=dict(
                s_feature=dict(
                    recorder='bb_l3_1_bn3',
                    from_student=True,
                    connector='loss_s3_sfeat'),
                t_feature=dict(recorder=['bb_l3_2_bn3',
                                         'bb_l3_3_bn3',
                                         'bb_l3_4_bn3',
                                         'bb_l3_5_bn3',
                                         'bb_l3_6_bn3',
                                         'bb_l3_7_bn3',
                                         'bb_l3_8_bn3',
                                         'bb_l3_9_bn3',
                                         'bb_l3_10_bn3',
                                         'bb_l3_11_bn3',
                                         'bb_l3_12_bn3',
                                         'bb_l3_13_bn3',
                                         'bb_l3_14_bn3',
                                         'bb_l3_15_bn3',
                                         'bb_l3_16_bn3',
                                         'bb_l3_17_bn3',
                                         'bb_l3_18_bn3',
                                         'bb_l3_19_bn3',
                                         'bb_l3_20_bn3',
                                         'bb_l3_21_bn3',
                                         'bb_l3_22_bn3',
                                         'bb_l3_23_bn3',
                                         'bb_l3_24_bn3',
                                         'bb_l3_25_bn3',
                                         'bb_l3_26_bn3'], from_student=False)),
            loss_4=dict(
                s_feature=dict(
                    recorder='bb_l4_1_bn3',
                    from_student=True,
                    connector='loss_s4_sfeat'),
                t_feature=dict(recorder='bb_l4_2_bn3', from_student=False)))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='exkd_uperhead_convnext-b_fcn_r18-80k'),
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
