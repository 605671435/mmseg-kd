_base_ = './shufflenet-v1-d8_fcn_1xb2-80k_synapse-512x512.py'

model = dict(backbone=dict(type='ShuffleNetV1_KD'))
