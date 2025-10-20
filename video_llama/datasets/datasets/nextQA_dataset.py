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
        
        super().__init__(vis_processor=vis_processor, audio_processor=None, text_processor=text_processor)


        # 读取一个路径下所有的

        df = pd.read_csv(os.path.join(ann_root, "text_data.csv"))
        df_example = pd.read_csv(os.path.join(ann_root, "text_data_examples.csv"))
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
        self.annotation_examples = df_example
        self.vis_root = vis_root
        self.num_frm = 8
        self.example_1_idx = 0
        self.example_2_idx = 1
        self.example_3_idx = 2

        self.example_question_prompt_1 = f'''{self._vqa_question(self.example_1_idx, is_example=True)}'''
        self.example_question_prompt_2 = f'''{self._vqa_question(self.example_2_idx, is_example=True)}'''
        self.example_question_prompt_3 = f'''{self._vqa_question(self.example_3_idx, is_example=True)}'''
        self.video_example_1 = self._get_video_examples(self.example_1_idx)
        self.video_example_2 = self._get_video_examples(self.example_2_idx)
        self.video_example_3 = self._get_video_examples(self.example_3_idx)


    def _get_video_path(self, idx, is_example):
        if is_example:
            sample = self.annotation_examples.iloc[idx]
        else:    
            sample = self.annotation.iloc[idx]
        sample_dict = sample.to_dict()        
        rel_video_fp = str(sample_dict['video']) + '.mp4'
        full_video_fp = os.path.join(self.video_des_folder,  rel_video_fp)
        return full_video_fp



    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            
            # sys_prompt = '''You are given video_embeddings, a question, and five options of answers to the question indexed by A, B, C, D, E. Your task is to select the correct answer to the question from the five options according to the video_embeddings.'''
            
            sys_prompt = '''You are given video embeddings, a question, and five options of answers to the question which are indexed by A, B, C, D, E. Your task is to select the correct answer index from the five options of answers to the question according to the video embeddings.'''
            question_prompt, correct_answer = self._vqa_question(index, is_example=False)

            sys_prompt = f'''<s>[INST]<<SYS>>{sys_prompt}<</SYS>>'''
            question_prompt_a = f'''Example: \nvideo_embeddings: '''
            question_prompt_a_real = f'''<Video>'''
            question_prompt_b = f'''</Video>{question_prompt}[/INST]'''

            # fetch video
            video_path = self._get_video_path(index, is_example=False) 
            # if os.path.exists(video_path):
            try:
                video = self.vis_processor(video_path)
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            ### to do:
            ### apply text processor on the prompts and text examples
            ### /to do
            
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
            "text_sys": sys_prompt,
            "text_a": question_prompt_a,
            "text_a_real": question_prompt_a_real, 
            "text_b": question_prompt_b,
            "text_e1": self.example_question_prompt_1,
            "text_e2": self.example_question_prompt_2,
            "text_e3": self.example_question_prompt_3,
            "video_e1": self.video_example_1,
            "video_e2": self.video_example_2,
            "video_e3": self.video_example_3,
            "correct_ans": correct_answer,
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)


    def _vqa_question(self, idx, is_example=False):
        if is_example:
            sample = self.annotation_examples.iloc[idx]
        else:
            sample = self.annotation.iloc[idx]
        
        sample_dict = sample.to_dict()
        question = sample_dict['question']
        answer_choices = []
        for i in range(5):
            answer_choices.append(sample_dict[f"a{i}"])
        
        correct_answer_idx = 'a' + str(sample_dict['answer'])
        correct_answer = sample_dict[correct_answer_idx]
        if is_example:
            correct_answer_in_example = sample_dict[correct_answer_idx]
        else:
            correct_answer_in_example = ""
        
        question_prompt = f'''qusetion: {question} options of answers: A.{answer_choices[0]} B.{answer_choices[1]} C.{answer_choices[2]} D.{answer_choices[3]} E.{answer_choices[4]} correct answer: {correct_answer_in_example}'''
        if is_example:
            return question_prompt
        else:
            return question_prompt, correct_answer
        
    def _get_video_examples(self, idx):
        video_path = self._get_video_path(idx, is_example=True) 
        # if os.path.exists(video_path):
        try:
            video = self.vis_processor(video_path)
            return video
        except:
            print(f"Failed to load examples with video: {video_path}.")
        
    
    # def collater(self, samples):
    #     new_result = {}
    #     new_result['image'] = default_collate( [sample["image"] for sample in samples])
    #     new_result['text_input'] = default_collate( [sample["text_input"] for sample in samples])
    #     return new_result
