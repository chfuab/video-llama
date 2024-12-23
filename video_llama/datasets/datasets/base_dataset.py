"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate


class BaseDataset(Dataset):
    """ def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        
        # vis_root (string): Root directory of images (e.g. coco/images/)
        # ann_root (string): directory to store the annotation file
        
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids() """
    
    def __init__(
        self, vis_processor=None, text_processor=None, audio_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r"))["annotations"])
            # self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
        self.annotation_mod = []
        for ann_path in ann_paths:
            self.annotation_mod.extend(json.load(open(ann_path, "r")))
        print(f"\n\n\n {ann_paths}, {self.annotation_mod}\n\n\n")

        self.vis_processor = vis_processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        # return len(self.annotation)
        return len(self.annotation_mod)

    def collater(self, samples):
        return default_collate(samples)

    """ def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor """

    def set_processors(self, vis_processor, audio_processor, text_processor):
        self.vis_processor = vis_processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor


    def _add_instance_ids(self, key="instance_id"):
        # for idx, ann in enumerate(self.annotation):
        for idx, ann in enumerate(self.annotation_mod):
            ann[key] = str(idx)


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
