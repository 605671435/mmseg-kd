from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.architectures.connectors import TorchFunctionalConnector
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from mmrazor.models.losses import ChannelWiseDivergence

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
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_cwd=dict(type=ChannelWiseDivergence, tau=4, loss_weight=3)),
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        connectors=dict(
            preds_S=dict(type=TorchFunctionalConnector,
                         function_name='interpolate',
                         func_args=dict(size=64))),
        loss_forward_mappings=dict(
            loss_cwd=dict(
                preds_S=dict(from_student=True, recorder='logits', connector='preds_S'),
                preds_T=dict(from_student=False, recorder='logits')))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='lits17', name='cwd_convnext_b_fcn_r18-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
