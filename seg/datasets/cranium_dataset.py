import os
from typing import List
import os.path as osp
import mmengine
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class CraniumDataset(BaseSegDataset):
    """CraniumDataset dataset.

    In segmentation map annotation for CraniumDataset,
    0 stands for background, which is included in 2 categories.
    ``reduce_zero_label`` is fixed to False. The ``img_suffix``
    is fixed to '.png' and ``seg_map_suffix`` is fixed to '.png'.

    Args:
        img_suffix (str): Suffix of images. Default: '.png'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default to False.
    """
    METAINFO = dict(classes=('background', 'hemorrhage'))

    def __init__(self,
                 case_list=None,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        self.case_list = case_list
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if hasattr(self, 'case_list'):
            lines = self.case_list
        else:
            if osp.isfile(self.ann_file):
                lines = mmengine.list_from_file(
                    self.ann_file, backend_args=self.backend_args)
            else:
                lines = os.listdir(img_dir)

        case_nums = dict()

        lines.sort()

        for line in lines:
            case_name = line.strip()
            imgs = os.listdir(osp.join(img_dir, case_name))
            imgs.sort()
            case_nums[case_name] = len(imgs)
            for img in imgs:
                data_info = dict(img_path=osp.join(img_dir, case_name, img))
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path'] = osp.join(ann_dir, case_name, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []

                data_info['case_name'] = case_name
                data_list.append(data_info)

        self._metainfo.update(case_nums=case_nums)
        if self._indices is not None and self._indices > 0:
            return data_list[:self._indices]
        else:
            return data_list
