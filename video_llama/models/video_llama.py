import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from video_llama.common.registry import registry
from video_llama.models.blip2 import Blip2Base, disabled_train
from video_llama.models.modeling_llama import LlamaForCausalLM, LlamaForSequenceClassification
# from video_llama.models.Qformer import BertEncoder
from transformers import LlamaTokenizer,BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
import copy
from video_llama.models.Qformer import BertConfig, BertLMHeadModel
from video_llama.models.ImageBind.models.imagebind_model import ImageBindModel,ModalityType
from video_llama.models.ImageBind.models import imagebind_model
# from flamingo_pytorch import PerceiverResampler
# import QformerAligned
from video_llama.models.Qformer_aligned import QformerAligned

@registry.register_model("video_llama")
class VideoLLAMA(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
        "pretrain_llama_v2": "configs/models/video_llama.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers =2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.

        frozen_llama_proj=True,
        frozen_video_Qformer=True,
        frozen_audio_Qformer=True,
        frozen_linear_proj=True,

        llama_proj_model='',
        fusion_header_type= "seqTransf",
        max_frame_pos= 32,
        fusion_head_layers = 2,
        num_video_query_token = 32,
        num_audio_query_token = 8,
        imagebind_ckpt_path = '/mnt/workspace/ckpt',
        equip_audio_branch = True,
        equip_QFormerAligned = False,
        # to-do: define qa_ckpt_path
        qa_ckpt_path = "/home/chfuab/LLM/video-llama/Video-LLaMA/ckpt",
        llama_type = "causalLM",
        use_lora = True,
    ):
        super().__init__()
        
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        if equip_QFormerAligned:
            logging.info("Loading pretrained QFormerAligned")
            self.q_former_aligned = QformerAligned(
                vit_model=vit_model,
                q_former_model=q_former_model,
                freeze_qformer=True,
                num_query_token=32,
                img_size=224,
                drop_path_rate=0,
                freeze_vit=True,
                use_grad_checkpoint=False,
                vit_precision="fp16",
                equip_audio_branch = True,
                imagebind_ckpt_path = '/mnt/workspace/ckpt'
            )
            # Frozen q_former_aligned
            for name, param in self.q_former_aligned.named_parameters():
                param.requires_grad = False
            self.q_former_aligned.eval()
            # Load the checkpoint for q_former_aligned
            assert (qa_ckpt_path), \
                "You must define QFormerAligned checkpoint if equip_QFormerAligned is True"
            q_former_aligned_ckpt = torch.load(qa_ckpt_path)
            self.q_former_aligned.load_state_dict(q_former_aligned_ckpt['model'], strict=False)

        self.equip_QFormerAligned = equip_QFormerAligned
        logging.info("Load pretrained QFormerAligned")

        logging.info('Loading linear projection layer after Q-Former')
        self.linear_proj = nn.Linear(self.query_tokens.size()[-1], 1024)        ################################
        if frozen_linear_proj:
            #  todo frozen  llama_proj
            for name, param in self.linear_proj.named_parameters():
                param.requires_grad = False
            logging.info('Linear proj is frozen')
        else:
            for name, param in self.linear_proj.named_parameters():
                param.requires_grad = True
            logging.info('Linear proj is not frozen')
        logging.info('Loading linear projection layer Done')

        logging.info('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token 
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.llama_tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]

        self.llama_type = llama_type
        logging.info('Loading LLAMA Model')
        if self.low_resource:
            print('low_resource')
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            print("normal resource")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLAMA Done')

        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        logging.info('Loading llama_proj Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)

        self.num_video_query_token = num_video_query_token
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,\
            vision_width=self.Qformer.config.hidden_size, num_hidden_layers =2)

        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None


        if frozen_video_Qformer:
            #  todo frozen  llama_proj
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            
            logging.info('video_Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            logging.info('video_Qformer is not frozen')

        if frozen_video_Qformer and (not frozen_audio_Qformer):
            self.train_flag = 1 # 只训练audio_Qformer
        elif not(frozen_video_Qformer) and frozen_audio_Qformer:
            self.train_flag = 0 # 训练video_Qformer
        elif not(frozen_video_Qformer) and not(frozen_audio_Qformer):
            self.train_flag = 2 # video_Qformer and AL trained
        else:
            self.train_flag = 3
        
        
        if frozen_video_Qformer and (not frozen_audio_Qformer) and (not frozen_linear_proj):
            self.train_VA_combined = True
        else:
            self.train_VA_combined = False

        if equip_audio_branch:
            print (f'Initializing audio encoder from {imagebind_ckpt_path} ...')
            self.audio_encoder,self.audio_hidden_size = \
                imagebind_model.imagebind_huge()
            torch.cuda.empty_cache()
            self.audio_encoder.load_state_dict(torch.load("{}/imagebind_huge.pth".format(imagebind_ckpt_path), map_location="cpu"))
            # free vision encoder
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            print ('audio encoder initialized.')
            
            self.num_audio_query_token = num_audio_query_token
            self.audio_Qformer,self.audio_query_tokens = self.init_video_Qformer(num_query_token = self.num_audio_query_token,\
                vision_width=self.audio_hidden_size, num_hidden_layers =2)
            self.audio_Qformer.cls = None
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.audio_llama_proj = nn.Linear(
                self.audio_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            self.audio_position_embedding = nn.Embedding(8, self.audio_hidden_size)
            self.audio_visual_position_embedding = nn.Embedding(8, self.audio_hidden_size)

            if frozen_audio_Qformer:
                #  todo frozen  llama_proj
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = False
                self.audio_query_tokens.requires_grad = False
                for name, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = False
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = False
                logging.info('audio_Qformer and audio-LLAMA proj is frozen')
            else:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = True
                self.audio_query_tokens.requires_grad = True
                for name, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = True
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = True
                logging.info('audio_Qformer is not frozen')
            if use_lora:
                for layer in self.audio_Qformer.bert.encoder.layer:
                    for name, param in layer.intermediate_query.lora.named_parameters():
                        param.requires_grad = True
                    for name, param in layer.output_query.lora.named_parameters():
                        param.requires_grad = True

        self.num_query_token = num_query_token

        #  self.audio_hidden_size
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_videoQformer_visual(self, image):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # frame attention
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama, atts_llama
    
    
    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            # print(prompt)
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img



    def encode_videoaudioQformer(self, 
                                 audio, 
                                 image, 
                                 audio_clip_times_all, 
                                 image_frame_idx_all, 
                                 video_frms_time_idx_all, 
                                 modality_type=ModalityType.AUDIO):
        # video_frms_time_idx_all: map of all video frame indices to their frame time 
        # image_frame_idx_all: map of sampled video frames indices to their time
        device_A = audio.device
        device_V = image.device

        batch_size, _, time_length_V, _, _ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
            
        with self.maybe_autocast():
            if not self.equip_QFormerAligned:
                image_embeds = self.ln_vision(self.visual_encoder(image)).to(device_V)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device_V)
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                q_hidden_state = query_output.last_hidden_state
                q_hidden_state = self.linear_proj(q_hidden_state)
            else:
                q_hidden_state = self.q_former_aligned(images=image)["q_former_branch_outputs"]

            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio, modality_type=modality_type)
            batch_size_A, time_length_A = audio.size()[:2]

        # finding position_id and calculate position embeddings
            position_embedding_A_all_list = []
            position_embedding_V_all_list = []
            audio_imagebind_finalout_k = []

            """ if not torch.is_tensor(image_frame_idx_all):
                image_frame_idx_all = torch.tensor(image_frame_idx_all)

            if image_frame_idx_all.dim() < 3:
                image_frame_idx_all = image_frame_idx_all.unsqueeze(0) """

            for k in range(batch_size_A):
                visual_audio_time_map = dict()
                for j, clip_time_tuple in enumerate(audio_clip_times_all[k]):
                    visual_audio_idx_lst = []
                    for pair in image_frame_idx_all[k]:
                        if pair[1] >= clip_time_tuple[0] and pair[1] <= clip_time_tuple[1]:
                            visual_audio_idx_lst.append(pair[0])
                        else:
                            continue
                    visual_audio_time_map[j] = visual_audio_idx_lst

            # calculate audio position_id
                position_id_list_A = []
                audio_imagebind_finalout_k_j = []
                for j, v_a_lst in visual_audio_time_map.items():
                    position_id_list_A.extend(v_a_lst)
                # dimension of audio_imagebind_finalout must be the same as position_id_list_A
                    temp = audio_imagebind_finalout[k, j, :].unsqueeze(0).expand(len(v_a_lst), -1)
                    audio_imagebind_finalout_k_j.append(temp)
                audio_imagebind_finalout_k_j_temp = torch.cat(audio_imagebind_finalout_k_j, dim=0)

                audio_imagebind_finalout_k.append(audio_imagebind_finalout_k_j_temp)

            # calculate audio position embedding
                position_id_list_A = [t.int() for t in position_id_list_A]
                position_id_list_A = torch.LongTensor(position_id_list_A).to(device_V)
                print(f"\n\n\n position_id_list_A: {position_id_list_A} \n\n\n")
                position_embedding_A = self.audio_position_embedding(position_id_list_A)  
                position_embedding_A_all_list.append(position_embedding_A)
            # calculate visual position embedding:
            
                position_id_img_frms = torch.LongTensor([pair[0].int() for pair in image_frame_idx_all[k]]).to(device_V)

                print(f"\n\n\n position_id_img_frms: {position_id_img_frms} \n\n\n")

                position_embed_V = self.audio_visual_position_embedding(position_id_img_frms)
                position_embedding_V_all_list.append(position_embed_V)

            audio_imagebind_finalout = torch.stack(audio_imagebind_finalout_k, dim=0) # use stack instead of cat!!?
            position_embedding_A_all = torch.stack(position_embedding_A_all_list, dim=0)
            position_embedding_V_all = torch.stack(position_embedding_V_all_list, dim=0)

        # combine audio_imagebind_finalout & frame_hidden_state with position embedding
            position_embedding_V_all = position_embedding_V_all.unsqueeze(-2)
            hidden_state_V = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length_V)
            hidden_state_V = position_embedding_V_all + hidden_state_V

            # Dimension of audio_imagebind_finalout must be same as position_embedding_A_all

            audio_imagebind_finalout = audio_imagebind_finalout + position_embedding_A_all
            
            hidden_state_V =  einops.rearrange(hidden_state_V, 'b t q h -> b (t q) h',b=batch_size,t=time_length_V)

        # Calculate encoder hidden state attendion
            frame_atts_A = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device_A)
            frame_atts_V = torch.ones(hidden_state_V.size()[:-1], dtype=torch.long).to(device_V)

        # Calculate query tokens
            # query_tokens_A = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
            query_tokens_all = self.video_query_tokens.expand(hidden_state_V.shape[0], -1, -1)

        # Calculate concatenated hidden states & attention mask
            hidden_states_all = torch.cat((hidden_state_V, audio_imagebind_finalout), dim=1)
            frame_atts_all = torch.cat((frame_atts_V, frame_atts_A), dim=1)

        # Put all things into Qformer. Assume 32 query tokens, hidden_state_V & audio_imagebind_finalout have same dim
            video_query_output = self.video_Qformer.bert(
                query_embeds=query_tokens_all, #[32,768]
                encoder_hidden_states=hidden_states_all,
                encoder_attention_mask=frame_atts_all,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
    
        return inputs_llama, atts_llama

    #  input audio shape [b t c h w] 
    def encode_audioQformer(self, audio,modality_type=ModalityType.AUDIO):
        device = audio.device
        with self.maybe_autocast():
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=modality_type)
            batch_size,time_length = audio.size()[:2]


            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            audio_position_embeddings = self.audio_position_embedding(position_ids)
            audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

            audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
            frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens, #[32,768]
                encoder_hidden_states=audio_imagebind_finalout,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            audio_hidden = audio_query_output.last_hidden_state

            inputs_llama = self.audio_llama_proj(audio_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
    
        return inputs_llama, atts_llama

    def encode_videoQformer_audiovideo(self, image, audio):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # encode audio 
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=ModalityType.AUDIO) # [batch,8*1,768]    8*32, 768
            audio_frame_position_embeddings = frame_position_embeddings.squeeze(-2)
            audio_feature = audio_feature + audio_frame_position_embeddings

            # frame attention a
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_hidden_state = torch.cat([frame_hidden_state,audio_feature],dim = 1)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens, #[32,768]
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
    
        return inputs_llama, atts_llama

    def forward(self, samples):
        if 'conv_type' in samples.keys() and samples['conv_type']=='multi':
            
            im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
            image = samples["images"]
            input_ids = samples['input_ids']
            if len(image.size())==4:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)

            if self.train_flag == 0:
                num_patch_tokens = self.num_video_query_token
                img_embeds, atts_img = self.encode_videoQformer_visual(image)
            elif self.train_flag == 1:
                num_patch_tokens = self.num_audio_query_token
                image = einops.rearrange(image, 'b c t h w -> b t c h w')
                img_embeds, atts_img = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
                
            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[temp_input_ids == im_patch_token_id] = 0
            temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)

            new_input_embeds=[]
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
                cur_image_features = img_embeds[cur_image_idx]

                if (cur_input_ids == im_patch_token_id).sum() != num_patch_tokens:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patch_tokens, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                
                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patch_tokens:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                
                cur_image_idx+=1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            targets = samples['labels']
            attention_mask = samples['attention_mask']
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            return {"loss": loss}
        else:
            image = samples["image"]
            audio = samples["audio"]    #
            audio_clip_times_all = samples["all_audio_clips_timepoints"]
            image_frame_idx_all = samples["image_idx_time_pair"]
            frms_time_idx_all = samples["all_idx_time_pair"]

            if len(image.size()) != 5:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)
            
            """ if self.train_flag == 1:
                image = einops.rearrange(image, 'b c t h w -> b t c h w')
                img_embeds, atts_img = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
                
            elif self.train_flag == 0:
                img_embeds, atts_img = self.encode_videoQformer_visual(image) """
            if self.train_VA_combined:
                img_embeds, atts_img = self.encode_videoaudioQformer(audio, image, audio_clip_times_all, image_frame_idx_all, frms_time_idx_all)    #
                

            if self.prompt_list:
                prompt = random.choice(self.prompt_list)
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
                

            self.llama_tokenizer.padding_side = "right"

            text = [t + self.end_sym for t in samples["text_input"]]

            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(image.device)

            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                        dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                            dtype=to_regress_tokens.input_ids.dtype,
                            device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.llama_model.model.embed_tokens(bos)
            atts_bos = atts_img[:, :1]

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            logits = outputs.logits
            
        return {"loss": loss, "logits": logits}

    """     
    def forward(self, samples):
        if 'conv_type' in samples.keys() and samples['conv_type']=='multi':
            
            im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
            image = samples["images"]
            input_ids = samples['input_ids']
            if len(image.size())==4:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)

            if self.train_flag == 0:
                num_patch_tokens = self.num_video_query_token
                img_embeds, atts_img = self.encode_videoQformer_visual(image)
            elif self.train_flag == 1:
                num_patch_tokens = self.num_audio_query_token
                image = einops.rearrange(image, 'b c t h w -> b t c h w')
                img_embeds, atts_img = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
                
            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[temp_input_ids == im_patch_token_id] = 0
            temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)

            new_input_embeds=[]
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
                cur_image_features = img_embeds[cur_image_idx]

                if (cur_input_ids == im_patch_token_id).sum() != num_patch_tokens:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patch_tokens, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                
                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patch_tokens:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                
                cur_image_idx+=1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            targets = samples['labels']
            attention_mask = samples['attention_mask']
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            return {"loss": loss}
        else:
            image = samples["image"]

            if len(image.size()) != 5:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)
            
            if self.train_flag == 1:
                image = einops.rearrange(image, 'b c t h w -> b t c h w')
                img_embeds, atts_img = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
            else:
                img_embeds, atts_img = self.encode_videoQformer_visual(image)

            if self.prompt_list:
                prompt = random.choice(self.prompt_list)
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
                

            self.llama_tokenizer.padding_side = "right"

            text = [t + self.end_sym for t in samples["text_input"]]

            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(image.device)

            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                        dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                            dtype=to_regress_tokens.input_ids.dtype,
                            device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.llama_model.model.embed_tokens(bos)
            atts_bos = atts_img[:, :1]

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss

        return {"loss": loss} 
    """


    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        
        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)
        frozen_audio_Qformer = cfg.get("frozen_audio_Qformer", True)
        frozen_linear_proj = cfg.get("frozen_linear_proj", True)

        llama_proj_model = cfg.get("llama_proj_model", '')
        
        fusion_header_type = cfg.get("fusion_header_type", 'seqTransf')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        fusion_head_layers = cfg.get("fusion_head_layers", 2)
        num_video_query_token =  cfg.get("num_video_query_token", 32)

        equip_audio_branch= cfg.get("equip_audio_branch", True)
        num_audio_query_token =  cfg.get("num_audio_query_token", 8)
        imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", '/mnt/workspace/ckpt')
        
        # Get config for QFormerAligned:
        equip_QFormerAligned = cfg.get("equip_QFormerAligned", False)
        qa_ckpt_path = cfg.get("qa_ckpt_path", "/home/chfuab/LLM/video-llama/Video-LLaMA/ckpt")
        llama_type = cfg.get("llama_type", "causalLM")
        use_lora = cfg.get("use_lora", True)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            fusion_header_type=fusion_header_type,
            max_frame_pos=max_frame_pos,
            fusion_head_layers=fusion_head_layers,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            frozen_linear_proj=frozen_linear_proj,
            num_video_query_token=num_video_query_token,
            num_audio_query_token = num_audio_query_token,
            imagebind_ckpt_path = imagebind_ckpt_path,
            equip_audio_branch = equip_audio_branch,
            llama_proj_model = llama_proj_model,
            equip_QFormerAligned = equip_QFormerAligned,
            qa_ckpt_path = qa_ckpt_path,
            llama_type = llama_type,
            use_lora=use_lora
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        ckpt_path_2 = cfg.get("ckpt_2", "")  
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model
    
    """ @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        
        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)
        frozen_audio_Qformer = cfg.get("frozen_audio_Qformer", True)

        llama_proj_model = cfg.get("llama_proj_model", '')
        
        fusion_header_type = cfg.get("fusion_header_type", 'seqTransf')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        fusion_head_layers = cfg.get("fusion_head_layers", 2)
        num_video_query_token =  cfg.get("num_video_query_token", 32)

        equip_audio_branch= cfg.get("equip_audio_branch", True)
        num_audio_query_token =  cfg.get("num_audio_query_token", 8)
        imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", '/mnt/workspace/ckpt')
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            fusion_header_type=fusion_header_type,
            max_frame_pos=max_frame_pos,
            fusion_head_layers=fusion_head_layers,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            num_video_query_token=num_video_query_token,
            num_audio_query_token = num_audio_query_token,
            imagebind_ckpt_path = imagebind_ckpt_path,
            equip_audio_branch = equip_audio_branch,
            llama_proj_model = llama_proj_model
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        ckpt_path_2 = cfg.get("ckpt_2", "")  
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model """

    def decode_llama_text(self, logits):
        # logits of shape (batch_size, seq_length, vocab_size)
        batch_size = logits.size()[0]
        seq_length = logits.size()[1]

        softmax_func = nn.Softmax(dim=2)
        result_ids = softmax_func(logits).argmax(dim=2)

        decoded_tokens = [[self.llama_tokenizer._convert_id_to_token(result_ids[k][s]) for s in range(seq_length)] for k in range(batch_size)]
        result_texts = {k: [self.llama_tokenizer.convert_tokens_to_string(decoded_tokens[k])] for k in range(batch_size)}
        return result_texts
    
    def decode_llama_choice(logits):
        return