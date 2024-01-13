# 基于知识蒸馏的医学图像语义分割库

一个基于知识蒸馏的医学图像语义分割库。

# 目录

- [安装要求](#安装要求)
    - [MMCV](#mmcv)
    - [其他安装包](#其他安装包)
- [数据集](#数据集)
    - [Synapse](#synapse)
- [训练和测试](#训练和测试)
- [方法库](#方法库)
    - [医学图像语义分割](#医学图像语义分割)
    - [知识蒸馏](#知识蒸馏)
- [参考文档](#参考文档)
- [致谢](#致谢)
- [引用](#引用)
- [License](#license)

# 安装要求
- Pytorch >= 1.12.0
- MMCV
## MMCV
该库依赖于 MMCV, MMCV 可以通过以下命令安装:

```pycon
pip install -U openmim
mim install mmcv
```

## 其他安装包
通过以下命令安装需要的安装包：

```pycon
pip install -r requirements.txt
```

# 数据集

## Synapse

- 官方网址 (未处理的3D原始数据，需要注册账号)：
https://www.synapse.org/#!Synapse:syn3193805/wiki/217752
- Google云盘链接 (**预处理好的2D切片数据，建议下载**)：
https://drive.google.com/file/d/1pNHGzZpCae-AjpswEFOUUWn5YrH7v66J/view?usp=drive_link

# 训练和测试

-   训练命令:

```pycon
python train.py {config}
```

-   测试命令:

```pycon
python test.py {config}
```
{config} 表示配置文件的路径. 配置文件可以在中 [configs](configs "configs")找到.
# 方法库

## 医学图像语义分割

- 常用的语义分割方法包括 FCN, U-Net, MedNeXT, MISSFormer, Swin-UNETR, TransU-Net
- 主要配置文件：

| 方法              |  Config |
| ------------------- |  ------ |
| FCN-R18            |  [config](configs/fcn/fcn_r18_d8_40k_synapse.py)      |
| FCN-R50            |  [config](configs/fcn/fcn_r50_d8_40k_synapse.py)      |
| UNet-R18            |  [config](configs/unet/unet_r18v1c_d8_40k_synapse.py)      |
| UNet-R50            |  [config](configs/unet/unet_r50_s4_d8_40k_synapse.py)      |
| Att-UNet            |  [config](configs/attn_unet/attn_ma_unet_r18v1c_synapse_40k.py)|
| MedNeXT             | [config](configs/medical_seg/mednext_40k_synapse.py)      |
| MISSFormer         | [config](configs/medical_seg/missformer_40k_synapse.py)      |
| TransU-Net          | [config](configs/medical_seg/transunet_40k_synapse.py)      |
| Swin-UNETR          | [config](configs/medical_seg/swin_unetr_base_40k_synapse.py)      |

### 参考文献

**FCN**
```bash
 @inproceedings{Long_Shelhamer_Darrell_2015,  
 title={Fully Convolutional Networks for Semantic Segmentation}, 
 url={http://dx.doi.org/10.1109/cvpr.2015.7298965}, 
 DOI={10.1109/cvpr.2015.7298965}, 
 booktitle={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
 author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor}, 
 year={2015}, 
 month={Jun}, 
 language={en-US} 
 }
```
**U-Net**
```bash
 @inbook{Ronneberger_Fischer_Brox_2015,  
 title={U-Net: Convolutional Networks for Biomedical Image Segmentation}, 
 url={http://dx.doi.org/10.1007/978-3-319-24574-4_28}, 
 DOI={10.1007/978-3-319-24574-4_28}, 
 booktitle={Lecture Notes in Computer Science,Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015}, 
 author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas}, 
 year={2015}, 
 month={Jan}, 
 pages={234–241}, 
 language={en-US} 
 }
```
**MedNeXT**
```bash
 @article{Roy_Koehler_Ulrich_Baumgartner_Petersen_Isensee_Jaeger_Maier-Hein,  
 title={MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation}, 
 author={Roy, Saikat and Koehler, Gregor and Ulrich, Constantin and Baumgartner, Michael and Petersen, Jens and Isensee, Fabian and Jaeger, PaulF and Maier-Hein, Klaus}, 
 language={en-US} 
 }
```
**MISSFormer**
```bash
 @article{Huang_Deng_Li_Yuan_2021,  
 title={MISSFormer: An Effective Medical Image Segmentation Transformer.}, 
 journal={Cornell University - arXiv,Cornell University - arXiv}, 
 author={Huang, Xiaotao and Deng, Zhifang and Li, Dandan and Yuan, Xueguang}, 
 year={2021}, 
 month={Sep}, 
 language={en-US} 
 }
```
**Swin-UNETR**
```bash
 @inbook{Hatamizadeh_Nath_Tang_Yang_Roth_Xu_2022,  
 title={Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images}, 
 url={http://dx.doi.org/10.1007/978-3-031-08999-2_22}, 
 DOI={10.1007/978-3-031-08999-2_22}, 
 booktitle={Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries,Lecture Notes in Computer Science}, 
 author={Hatamizadeh, Ali and Nath, Vishwesh and Tang, Yucheng and Yang, Dong and Roth, Holger R. and Xu, Daguang}, 
 year={2022}, 
 month={Jan}, 
 pages={272–284}, 
 language={en-US} 
 }
```
**TransU-Net**
```bash
 @article{Chen_Lu_Yu_Luo_Adeli_Wang_Lu_Yuille_Zhou_2021,  
 title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation}, 
 journal={Cornell University - arXiv,Cornell University - arXiv}, 
 author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, AlanL. and Zhou, Yuyin}, 
 year={2021}, 
 month={Feb}, 
 language={en-US} 
 }
```
## 知识蒸馏

- 知识蒸馏库位于 [configs/distll](configs/distll) 中，包含KD, AT, DKD, DIST, CWD, ReviewKD

| 方法              |  Config |
| ------------------- |  ------ |
| KD            |  [config](configs/distill/kd)      |
| AT            |  [config](configs/distill/at)|
| DKD          | [config](configs/distill/dkd)      |
| DIST          | [config](configs/distill/DIST)      |
| CWD          | [config](configs/distill/cwd)      |
| ReviewKD          | [config](configs/distill/reviewkd)      |
### 教师权重
知识蒸馏需要先**预训练教师**，这里提供**部分**教师权重：

Google云盘：https://drive.google.com/drive/folders/1E5aduWgQXbeFkMTROdfUpUl5icDWpMQI?usp=drive_link

建议自己训练教师权重。
### 参考文献

**KD**
```bash
 @article{Hinton_Vinyals_Dean_2015,  
 title={Distilling the Knowledge in a Neural Network}, 
 journal={arXiv: Machine Learning,arXiv: Machine Learning}, 
 author={Hinton, GeoffreyE. and Vinyals, Oriol and Dean, J.Michael}, 
 year={2015}, 
 month={Mar}, 
 language={en-US} 
 }
```
**AT**
```bash
 @article{Zagoruyko_Komodakis_2016,  
 title={Paying more attention to attention: improving the performance of convolutional neural networks via attention transfer}, 
 journal={Le Centre pour la Communication Scientifique Directe - HAL - Université Paris Descartes,Le Centre pour la Communication Scientifique Directe - HAL - Université Paris Descartes}, 
 author={Zagoruyko, Sergey and Komodakis, Nikos}, 
 year={2016}, 
 month={Nov}, 
 language={en-US} 
 }
```
**DKD**
```bash
 @article{Zhao_Cui_Song_Qiu_Liang,  
 title={Decoupled Knowledge Distillation}, 
 author={Zhao, Borui and Cui, Quan and Song, Renjie and Qiu, Yiyu and Liang, Jiajun}, 
 language={en-US} 
 }
```
**DIST**
```bash
 @article{Huang_You_Wang_Qian_Xu_2022,  
 title={Knowledge Distillation from A Stronger Teacher}, 
 author={Huang, Tao and You, Shan and Wang, Fei and Qian, Chen and Xu, Chang}, 
 year={2022}, 
 month={May}, 
 language={en-US} 
 }
```
**CWD**
```bash
 @article{Shu_Liu_Gao_Xu_Shen_2020,  
 title={Channel-wise Distillation for Semantic Segmentation.}, 
 author={Shu, Changyong and Liu, Yifan and Gao, Jianfei and Xu, Lin and Shen, Chunhua}, 
 year={2020}, 
 month={Nov}, 
 language={en-US} 
 }
```
**ReviewKD**
```bash
 @inproceedings{Chen_Liu_Zhao_Jia_2021,  
 title={Distilling Knowledge via Knowledge Review}, 
 url={http://dx.doi.org/10.1109/cvpr46437.2021.00497}, 
 DOI={10.1109/cvpr46437.2021.00497}, 
 booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
 author={Chen, Pengguang and Liu, Shu and Zhao, Hengshuang and Jia, Jiaya}, 
 year={2021}, 
 month={Jun}, 
 language={en-US} 
 }
```
# 参考文档

- 实验基础平台：[MMEngine](https://mmengine.readthedocs.io/zh-cn/latest/index.html)
- 语义分割平台：[MMSegmentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/)
- 知识蒸馏平台：[MMRazor](https://mmrazor.readthedocs.io/en/latest/)

# 致谢

特别致谢:
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation "MMSegmentation")
- [MMEngine](https://github.com/open-mmlab/mmengine "MMEngine")
- [Monai](https://github.com/Project-MONAI)
- [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt)
- [MISSFormer](https://github.com/ZhifangDeng/MISSFormer/tree/main)
- 江西理工大学-信息工程学院

# 引用

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
