"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
import shutil
class NextQATrainDataset(BaseDataset):
    """ def __init__(self, vis_processor, text_processor, vis_root, ann_root):
        
        # vis_root (string): Root directory of video (e.g. webvid_eval/video/)
        # ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        # split (string): val or test
        
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)


        # 读取一个路径下所有的

        ts_df = []
        for file_name in os.listdir(ann_root):
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(ann_root, file_name))
                ts_df.append(df)

        merged_df = pd.concat(ts_df)
        self.annotation = merged_df
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = 8
        self.frm_sampling_strategy = 'headtail' """

    def __init__(self, vis_processor, text_processor, audio_processor, vis_root, ann_root):
        
        # vis_root (string): Root directory of video (e.g. webvid_eval/video/)
        # ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        # split (string): val or test
        
        super().__init__(vis_processor=vis_processor, audio_processor=audio_processor, text_processor=text_processor)


        # 读取一个路径下所有的

        df = pd.read_csv(os.path.join(ann_root, "train.csv"))
        video_src_folder = os.path.join(vis_root, "nextqa-video")
        video_des_folder = os.path.join(vis_root, "nextqa-video-extracted")

        self.video_src_folder = video_src_folder
        self.video_des_folder = video_des_folder

        video_sub_folders = os.listdir(video_src_folder)
        for s in video_sub_folders:
            subfolder_path = os.path.join(video_src_folder, s)
            files = os.listdir(subfolder_path)
            for file in files:
                file_path = os.path.join(subfolder_path, file)
                des_file_path = os.path.join(video_des_folder, file)
                shutil.move(file_path, des_file_path)

        self.annotation = df
        self.vis_root = vis_root
        self.resize_size = 600     ###
        self.num_frm = 8
        self.frm_sampling_strategy = 'headtail' ###


    def _get_video_path(self, sample):
        rel_video_fp = str(sample['video']) + '.mp4'
        full_video_fp = os.path.join(self.video_des_folder,  rel_video_fp)
        return full_video_fp



    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            sample = self.annotation.iloc[index]
            sample_dict = sample.to_dict()
            video_id = sample_dict['video']

            question = sample_dict['question']
            answer_choices = []
            for i in range(5):
                answer_choices.append(sample_dict[f"a{i}"])
            
            correct_answer_idx = 'a' + str(sample_dict['answer'])
            correct_answer = sample_dict[correct_answer_idx]

            sys_prompt = '''You are given video_embeddings, a question, and five options of answers to the question indexed by a0, a1, a2, a3, a4. Your task is to select the correct answer to the question according to the video_embeddings.'''
            
            question_prompt = f'''qusetion: {question}\nanswer options:\na0. {answer_choices[0]}\na1. {answer_choices[1]}\na2. {answer_choices[2]}\na3. {answer_choices[3]}\na4. {answer_choices[4]}\ncorrect answer: '''


            wrapped_sys_prompt = f'''<s>[INST] <<SYS>>
                        {sys_prompt}
                        <</SYS>>[/INST]</s>'''
            wrapped_question_prompt_a = f'''<s>[INST]'''
            wrapped_question_prompt_b = f'''{question_prompt}[/INST]'''

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

            # print(video.size())
            if video is None:
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
            "text_sys": wrapped_sys_prompt,
            "text_a": wrapped_question_prompt_a,
            "text_b": wrapped_question_prompt_b,
            "correct_ans": correct_answer,
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)

    # def collater(self, samples):
    #     new_result = {}
    #     new_result['image'] = default_collate( [sample["image"] for sample in samples])
    #     new_result['text_input'] = default_collate( [sample["text_input"] for sample in samples])
    #     return new_result
        
class WebvidDatasetEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)

        return {
            "video": video,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }


