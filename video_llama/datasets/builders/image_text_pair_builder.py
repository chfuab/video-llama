import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.laion_dataset import LaionDataset
from video_llama.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset


@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset
    eval_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = [build_info.storage_train, build_info.storage_eval]

        datasets = dict()
        splits = ["train", "eval"]
        
        for i in range(2): 
            if not os.path.exists(storage_path[i]):
                warnings.warn("storage path {} does not exist.".format(storage_path[i]))

        # create datasets
        dataset_cls = [self.train_dataset_cls, self.eval_dataset_cls]
        for i, split in enumerate(splits):
            datasets[split] = dataset_cls[i](
                vis_processor=self.vis_processors[split],
                text_processor=self.text_processors[split],
                ann_paths=[os.path.join(storage_path[i], 'metadata.json')],
                # ann_paths=[os.path.join(storage_path[i], 'filter_cap.json')],
                vis_root=os.path.join(storage_path[i], 'image'),
            )
        print(f"\n\n\n dataset: {datasets['train']}\n\n\n")
        return datasets

