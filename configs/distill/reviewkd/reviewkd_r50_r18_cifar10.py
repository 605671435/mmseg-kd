from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ReviewKDDistiller
from razor.models.losses import HCL
from razor.models.architectures.connectors import ABFConnector
from mmrazor.models.task_modules.recorder import ModuleInputsRecorder
from mmpretrain.visualization import UniversalVisualizer

with read_base():
    from ..._base_.datasets.cifar10 import *  # noqa
    from ..._base_.schedules.cifar10_bs128 import *  # noqa
    from ..._base_.default_runtime import *  # noqa
    # import teacher model and student model
    from ...resnet.resnet50_cifar10 import model as teacher_model  # noqa
    from ...resnet.resnet18_cifar10 import model as student_model  # noqa

teacher_ckpt = 'ckpts/resnet50_cifar10/top1_95-46_epoch_165.pth'  # noqa: E501

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
            bb_s1=dict(type=ModuleInputsRecorder, source='backbone.layer1.2.relu'),
            bb_s2=dict(type=ModuleInputsRecorder, source='backbone.layer2.3.relu'),
            bb_s3=dict(type=ModuleInputsRecorder, source='backbone.layer3.5.relu'),
            bb_s4=dict(type=ModuleInputsRecorder, source='backbone.layer4.2.relu')),
        distill_losses=dict(
            loss_hcl_s1=dict(type=HCL, loss_weight=1),
            loss_hcl_s2=dict(type=HCL, loss_weight=1),
            loss_hcl_s3=dict(type=HCL, loss_weight=1),
            loss_hcl_s4=dict(type=HCL, loss_weight=1)),
        connectors=dict(
           loss_s4_sfeat=dict(type=ABFConnector,
                              student_channels=512,
                              mid_channel=512,
                              teacher_channels=2048,
                              teacher_shapes=4,
                              student_shapes=4,
                              fuse=False),
           loss_s3_sfeat=dict(type=ABFConnector,
                              student_channels=256,
                              mid_channel=512,
                              teacher_channels=1024,
                              teacher_shapes=8,
                              student_shapes=8,
                              fuse=True),
           loss_s2_sfeat=dict(type=ABFConnector,
                              student_channels=128,
                              mid_channel=512,
                              teacher_channels=512,
                              teacher_shapes=16,
                              student_shapes=16,
                              fuse=True),
           loss_s1_sfeat=dict(type=ABFConnector,
                              student_channels=64,
                              mid_channel=512,
                              teacher_channels=256,
                              teacher_shapes=32,
                              student_shapes=32,
                              fuse=True)),
        loss_forward_mappings=dict(
            loss_hcl_s4=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s4',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s4_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s4', record_idx=2, data_idx=0)),
            loss_hcl_s3=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s3',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s3_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s3', record_idx=2, data_idx=0)),
            loss_hcl_s2=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s2',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s2_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s2', record_idx=2, data_idx=0)),
            loss_hcl_s1=dict(
                s_feature=dict(from_student=True,
                               recorder='bb_s1',
                               record_idx=1,
                               data_idx=0,
                               connector='loss_s1_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s1', record_idx=2, data_idx=0)))))

find_unused_parameters = True
log_processor = dict(by_epoch=True)
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='cifar10', name='reviewkd_r50_r18'))
]
visualizer = dict(type=UniversalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')

