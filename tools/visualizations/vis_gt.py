# from mmseg.visualization import SegLocalVisualizer
from mmcv.transforms import Compose
from seg.visualization.local_visualizer import SegLocalVisualizer
from mmseg.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
from mmseg.datasets.transforms.formatting import PackSegInputs
import mmengine
import mmcv
import os
import os.path as osp
from seg.datasets.synapse import SynapseDataset
from seg.datasets.flare22 import FLARE22Dataset
import argparse
from mmengine.utils import ProgressBar
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('img_path', help='train config file path')
    args = parser.parse_args()
    return args
def vis_gt(img_path, dataset_name='synapse'):
    if dataset_name == 'synapse':
        dataset = SynapseDataset
    elif dataset_name == 'flare22':
        dataset = FLARE22Dataset
    else:
        raise TypeError
    transforms = Compose([
        dict(type=LoadImageFromFile, color_type='grayscale'),
        dict(type=LoadAnnotations),
        dict(type=PackSegInputs)
    ])
    data = transforms({'img_path': img_path,
                       'seg_map_path': img_path.replace('img_dir', 'ann_dir').replace('jpg', 'png'),
                       'reduce_zero_label': None,
                       'seg_fields': []})
    visualizer = SegLocalVisualizer(alpha=1.0)
    visualizer.set_dataset_meta(classes=dataset.METAINFO['classes'],
                                palette=dataset.METAINFO['palette'],
                                dataset_name=dataset_name)

    img_bytes = mmengine.fileio.get(img_path)
    img = mmcv.imfrombytes(img_bytes)
    img = img[:, :, ::-1]
    visualizer.add_datasample(
                    osp.splitext(osp.basename(img_path))[0],
                    img,
                    data['data_samples'],
                    draw_gt=True,
                    draw_pred=False,
                    out_file=osp.join('./out_dir/gt', dataset_name, osp.basename(img_path)))
def main():
    # args = parse_args()
    # img_dir = '/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val'
    img_dir = '/home/jz207/workspace/zhangdw/ex_kd/data/FLARE22/img_dir/val'
    pb1 = ProgressBar(len(os.listdir(img_dir)))
    for case in os.listdir(img_dir):
        pb2 = ProgressBar(len(os.listdir(osp.join(img_dir, case))))
        for img in os.listdir(osp.join(img_dir, case)):
            vis_gt(osp.join(img_dir, case, img), dataset_name='flare22')
            pb2.update()
        pb1.update()
    # vis_gt(args.img_path)
if __name__=='__main__':
    main()
    # vis_gt('/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0001/case0001_slice094.jpg')