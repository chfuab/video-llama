import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.webvid_datasets import WebvidDataset
from video_llama.datasets.datasets.vast_27M_dataset import VAST27MDataset

@registry.register_builder("webvid")
class WebvidBuilder(BaseDatasetBuilder):
    train_dataset_cls = WebvidDataset
    eval_dataset_cls = WebvidDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/webvid/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()

        build_info = self.config.build_info
        dataset_cls = [self.train_dataset_cls, self.eval_dataset_cls]

        splits = ["train", "eval"]
        data_root_all = [[build_info.videos_dir, build_info.anno_dir], 
                         [build_info.videos_dir_val, build_info.anno_dir_val]]

        for i, split in enumerate(splits):
            datasets[split] = dataset_cls[i](
                vis_processor=self.vis_processors[split],
                audio_processor=self.audio_processor[split],
                text_processor=self.text_processors[split],
                vis_root=data_root_all[i][0],
                ann_root=data_root_all[i][1]
            )

        return datasets

    """ def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets """
    
@registry.register_builder("vast27M")
class Vast27MBuilder(BaseDatasetBuilder):
    train_dataset_cls = VAST27MDataset
    eval_dataset_cls = VAST27MDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vast27m/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()

        build_info = self.config.build_info
        dataset_cls = [self.train_dataset_cls, self.eval_dataset_cls]

        splits = ["train", "eval"]
        data_root_all = [[build_info.videos_dir, build_info.anno_dir], 
                         [build_info.videos_dir_val, build_info.anno_dir_val]]

        for i, split in enumerate(splits):
            datasets[split] = dataset_cls[i](
                vis_processor=self.vis_processors[split],
                audio_processor=self.audio_processor[split],
                text_processor=self.text_processors[split],
                vis_root=data_root_all[i][0],
                ann_root=data_root_all[i][1]
            )

        return datasets