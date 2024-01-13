from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ReviewKDDistiller
from razor.models.losses import HCL
from razor.models.architectures.connectors import ABFConnector
from mmrazor.models.task_modules.recorder import ModuleInputsRecorder, ModuleOutputsRecorder
from seg.engine.hooks import MyCheckpointHook

with read_base():
    from ..._base_.datasets.lits17 import *  # noqa
    from ..._base_.schedules.schedule_80k import *  # noqa
    from ..._base_.default_runtime import *  # noqa
    # import teacher model and student model
    from ...convnext.conv_next_b_list17_80k import model as teacher_model  # noqa
    from ...resnet.fcn_r18_no_pretrain_80k_lits17 import model as student_model  # noqa

teacher_ckpt = 'ckpts/conv_next_b-lits17-80k/best_mDice_82-28_iter_80000.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ReviewKDDistiller,
        student_recorders=dict(
            bb_s1=dict(type=ModuleInputsRecorder, source='backbone.layer1.1.relu'),
            bb_s2=dict(type=ModuleInputsRecorder, source='backbone.layer2.1.relu'),
            bb_s3=dict(type=ModuleInputsRecorder, source='backbone.layer3.1.relu'),
            bb_s4=dict(type=ModuleInputsRecorder, source='backbone.layer4.1.relu')),
        teacher_recorders=dict(
            bb_s1=dict(type=ModuleOutputsRecorder, source='backbone.stages.0.2'),
            bb_s2=dict(type=ModuleOutputsRecorder, source='backbone.stages.1.2'),
            bb_s3=dict(type=ModuleOutputsRecorder, source='backbone.stages.2.26'),
            bb_s4=dict(type=ModuleOutputsRecorder, source='backbone.stages.3.2')),
        distill_losses=dict(
            loss_hcl_s4=dict(type=HCL, loss_weight=0.1),
            loss_hcl_s3=dict(type=HCL, loss_weight=0.001),
            loss_hcl_s2=dict(type=HCL, loss_weight=0.01),
            loss_hcl_s1=dict(type=HCL, loss_weight=0.1)),
        connectors=dict(
           loss_s4_sfeat=dict(type=ABFConnector,
                              student_channels=512,
                              mid_channel=512,
                              teacher_channels=1024,
                              student_shapes=32,
                              teacher_shapes=8,
                              fuse=False),
           loss_s3_sfeat=dict(type=ABFConnector,
                              student_channels=256,
                              mid_channel=512,
                              teacher_channels=512,
                              student_shapes=32,
                              teacher_shapes=16,
                              fuse=True),
           loss_s2_sfeat=dict(type=ABFConnector,
                              student_channels=128,
                              mid_channel=512,
                              teacher_channels=256,
                              student_shapes=32,
                              teacher_shapes=32,
                              fuse=True),
           loss_s1_sfeat=dict(type=ABFConnector,
                              student_channels=64,
                              mid_channel=512,
                              teacher_channels=128,
                              student_shapes=64,
                              teacher_shapes=64,
                              fuse=True)),
        loss_forward_mappings=dict(
            loss_hcl_s4=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s4',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s4_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s4')),
            loss_hcl_s3=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s3',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s3_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s3')),
            loss_hcl_s2=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s2',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s2_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s2')),
            loss_hcl_s1=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s1',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s1_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s1')))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='lits17', name='reviewkd_convnext-b_fcn_r18-80k'),
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
