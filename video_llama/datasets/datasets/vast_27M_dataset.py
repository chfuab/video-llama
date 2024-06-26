import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
import json

class VAST27MDataset(BaseDataset):

    def __init__(self, vis_processor, text_processor, audio_processor, vis_root, ann_root):
        
        # vis_root (string): Root directory of video (e.g. webvid_eval/video/)
        # ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        # split (string): val or test
        
        super().__init__(vis_processor=vis_processor, audio_processor=audio_processor, text_processor=text_processor)


        # 读取一个路径下所有的

        ts_df = []
        for file_name in os.listdir(ann_root):
            if file_name.endswith('.json'):
                df = pd.read_json(os.path.join(ann_root, file_name), orient="records")
                ts_df.append(df)

        merged_df = pd.concat(ts_df)
        print(merged_df.to_string())
        self.annotation = merged_df
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = 8
        self.frm_sampling_strategy = 'visual-audio-aligned'


    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(str(sample['titles']) + '.mp4')
        full_video_fp = os.path.join(self.vis_root,  rel_video_fp)
        return full_video_fp

    def _get_audio_path(self, sample):
        rel_audio_fp = os.path.join(str(sample['titles']) + '.wav')
        full_audio_fp = os.path.join(self.vis_root,  rel_audio_fp)
        return full_audio_fp

    """ def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            sample = self.annotation.iloc[index]
            sample_dict = sample.to_dict()
            video_id = sample_dict['videoid']

            if 'name' in sample_dict.keys():
                text = sample_dict['name'].strip()
            else:
                raise NotImplementedError("Un-supported text annotation format.")

            # fetch video
            video_path = self._get_video_path(sample_dict) 
            # if os.path.exists(video_path):
            try:
                video = self.vis_processor(video_path)
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            caption = self.text_processor(text)

            # print(video.size())
            if video is None or caption is None \
                    or video.size()!=torch.Size([3,self.vis_processor.n_frms,224,224]):
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "text_input": caption,
            "type":'video',
        } """

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            sample = self.annotation.iloc[index]
            sample_dict = sample.to_dict()

            if 'vast_cap' in sample_dict.keys():
                text = sample_dict['vast_cap'].strip()
            else:
                raise NotImplementedError("Un-supported text annotation format.")

            # fetch video
            video_path = self._get_video_path(sample_dict)
            audio_path = self._get_audio_path(sample_dict)
            # if os.path.exists(video_path):

            try:
                audio, all_clips_timepoints_all = self.audio_processor(audio_path)
            except:
                print(f"Failed to load examples with audio from {audio_path}")            
            try:
                video, image_idx_time_pair, all_idx_time_pair = self.vis_processor(video_path, all_clips_timepoints_all)
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            caption = self.text_processor(text)

            # print(video.size())
            if video is None or caption is None or audio is None\
                    or video.size()!=torch.Size([3,self.vis_processor.n_frms,224,224]):
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "audio": audio,
            "all_audio_clips_timepoints": torch.tensor(all_clips_timepoints_all),
            "image_idx_time_pair": torch.tensor(image_idx_time_pair),
            "all_idx_time_pair": torch.tensor(all_idx_time_pair), 
            "text_input": caption,
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)