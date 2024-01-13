from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from mmrazor.models.losses import DKDLoss

with read_base():
    from ..._base_.datasets.cifar10 import *  # noqa
    from ..._base_.schedules.cifar10_bs128 import *  # noqa
    from ..._base_.default_runtime import *  # noqa
    # import teacher model and student model
    from ...resnet.resnet18_cifar10 import model as teacher_model  # noqa
    from ...resnet.resnet18_cifar10 import model as student_model  # noqa

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            fc=dict(type=ModuleOutputsRecorder, source='head.fc'),
            gt_labels=dict(type=ModuleInputsRecorder, source='head.loss_module')),
        teacher_recorders=dict(
            fc=dict(type=ModuleOutputsRecorder, source='head.fc')),
        distill_losses=dict(
            loss_dkd=dict(
                type=DKDLoss,
                tau=1,
                beta=0.5,
                loss_weight=1,
                reduction='mean')),
        loss_forward_mappings=dict(
            loss_dkd=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc'),
                gt_labels=dict(
                    recorder='gt_labels', from_student=True, data_idx=1)))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='dkd-unet-base-unet-small-1000e'),
        define_metric_cfg=dict(Dice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
