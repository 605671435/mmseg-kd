# Decoupled Pixel-wise Correlation for Abdominal CT Multi-organ Segmentation

A pytorch implenment for Decoupled Pixel-wise Correlation for Abdominal CT Multi-organ Segmentation

# Directory

- [Requirement](#requirement)
    - [MMCV](#mmcv)
    - [Other required package](#other-required-package)
- [Datasets](#datasets)
    - [Synapse](#synapse)
    - [FLARE22](#flare22)
- [Training and testing](#training-and-testing)
- [Results](#results)
    - [Synapse](#results-on-synapse)
    - [FLARE22](#results-on-flare22)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

# Requirement
- Pytorch >= 1.12.0
- MMCV
## MMCV
This repo is required the MMCV, it can be simply install by:

```pycon
pip install -U openmim
mim install mmcv
```

## Other required package
We have a lot of required package listed in [here](requirements.txt).

You can simply run the following script to install the required package:

```pycon
pip install -r requirements.txt
```

# Datasets

## Synapse

You counld get Synapse raw data from [here](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752 "https://www.synapse.org/#!Synapse:syn3193805/wiki/217752").

## FLARE22
You counld get FLARE22 raw data from [here](https://flare22.grand-challenge.org).

# Training and testing

-   Training on one GPU:

```pycon
python train.py {config}
```

-   Testing on one GPU:

```pycon
python test.py {config}
```
{config} means the config path. The config path can be found in [configs](new_configs "new_configs"),
or you can find them in the last column of the following tables.
# Results

## Results on Synapse

| Method              | DSC(%)         | JC (%)         | HD (mm)        | Iteration time (ms) | FLOPs&#xA;&#xA;(G) | Params&#xA;&#xA;(M) | Config |
| ------------------- | -------------- | -------------- | -------------- | ----------------------- | ------------------ | ------------------- | ------ |
| UNet-R18            | 73.46±0.41     | 62.80±0.35     | 54.60±4.07     | 64.1                    | 61.116             | 13.694              | [config](new_configs/unet/unet_r18v1c_d8_40k_synapse.py)      |
| Att-UNet            | 75.37±1.46     | 65.29±1.66     | 48.17±4.22     | 77.8                    | 64.964             | 15.076              | [config](new_configs/attn_unet/attn_ma_unet_r18v1c_synapse_40k.py)|
| Swin-UNETR          | 76.03±0.71     | 66.05±0.87     | 45.92±3.36     | 265.9                   | 71.535             | 25.138              | [config](new_configs/medical_seg/swin_unetr_base_40k_synapse.py)      |
| MISSFormer          | 77.39±0.73     | 67.48±0.84     | 40.12±3.15     | 374.9                   | 58.155             | 42.463              | [config](new_configs/medical_seg/missformer_40k_synapse.py)      |
| ConvNeXt            | 73.66±0.64     | 63.28±0.94     | 49.34±6.37     | 169.9                   | 113.486            | 98.145              | [config](new_configs/convnext/nopretrained_unet_conv_next_b_synapse_40k.py)      |
| SegNeXt             | 74.04±1.13     | 63.48±1.32     | 41.40±4.41     | 250.0                   | 33.831             | 28.854              | [config](new_configs/segnext/nopretrain_unet_segnext_b_40k_synapse.py)      |
| MedNeXt             | 77.05±0.35     | 67.18±0.41     | 38.46±4.49     | 244.9                   | 54.019             | 10.448              | [config](new_configs/medical_seg/mednext_40k_synapse.py)      |
| Ham-UNet            | 76.90±0.75     | 67.39±0.93     | 44.32±6.55     | 83.4                    | 66.925             | 14.908              | [config](new_configs/hamnet/unet_r18v1c_hamnet_40k_synapse.py)      |
| **DPCA-Net (ours)** | 77.50±0.30     | 67.82±0.47     | 40.52±4.65     | 68.0                    | 66.030             | 14.974              | [config](new_configs/dsnet/unet_r18v1c_dsnet_v14_dam_40k_synapse.py)      |
| **DPCS-Net (ours)** | **77.98±0.71** | **68.33±0.86** | **36.35±5.82** | 67.9                    | 65.963             | 14.991              | [config](new_configs/dsnet/unet_r18v1c_dsnet_v14_40k_synapse.py)      |

## Results on FLARE22

| Method              | DSC(%)         | JC (%)         | HD (mm)        | Config |
| ------------------- | -------------- | -------------- | -------------- | ------ |
| UNet-R18            | 85.63±0.56     | 76.95±0.81     | 16.16±1.65     | [config](new_configs/unet/unet_r18v1c_d8_40k_flare22.py)      |
| Att-UNet            | 86.09±0.75     | 77.60±1.00     | 14.04±1.25     | [config](new_configs/attn_unet/attn_ma_unet_r18v1c_flare22_40k.py)      |
| Swin-UNETR          | 85.72±0.25     | 77.05±0.28     | 17.06±1.86     | [config](new_configs/medical_seg/swin_unetr_base_40k_flare22.py)      |
| MISSFormer          | 84.88±0.19     | 75.68±0.27     | 15.14±0.55     | [config](new_configs/medical_seg/missformer_40k_flare22.py)      |
| ConvNeXt            | 84.45±0.43     | 75.20±0.57     | 17.66±2.70     | [config](new_configs/convnext/nopretrained_unet_conv_next_b_flare22_40k.py)      |
| SegNeXt             | 84.73±0.28     | 75.76±0.35     | 13.18±1.41     | [config](new_configs/segnext/nopretrain_unet_segnext_b_40k_flare22.py)      |
| MedNeXt             | 86.36±0.38     | 77.90±0.44     | 21.70±1.77     | [config](new_configs/medical_seg/mednext_40k_flare22.py)      |
| Ham-UNet            | 86.78±0.17     | 78.47±0.23     | 11.34±1.61     | [config](new_configs/hamnet/unet_r18v1c_hamnet_40k_flare22.py)      |
| **DPCA-Net (ours)** | 86.94±0.20     | 78.69±0.25     | 11.52±0.82     | [config](new_configs/dsnet/unet_r18v1c_dsnet_v14_dam_40k_flare22.py)      |
| **DPCS-Net (ours)** | **87.04±0.19** | **78.78±0.26** | **10.87±0.74** | [config](new_configs/dsnet/unet_r18v1c_dsnet_v14_40k_synapse.py)      |

# Acknowledgement

Specially thanks to the following:
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation "MMSegmentation")
- [MMEngine](https://github.com/open-mmlab/mmengine "MMEngine")
- [Monai](https://github.com/Project-MONAI)
- [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt)
- [MISSFormer](https://github.com/ZhifangDeng/MISSFormer/tree/main)

# Citation

```bash
@misc{mmseg2020,
  title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
  author={MMSegmentation Contributors},
  howpublished = {\url{[https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)}},
  year={2020}
}
```

```bash
@article{mmengine2022,
  title   = {{MMEngine}: OpenMMLab Foundational Library for Training Deep Learning Models},
  author  = {MMEngine Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmengine}},
  year={2022}
}
```

# License

This project is released under the [Apache 2.0 license](https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE "Apache 2.0 license") of mmsegmentation.
