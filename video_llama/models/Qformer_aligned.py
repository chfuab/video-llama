import random as rnd
import logging
import torch
import torch.nn as nn

from video_llama.models.blip2 import Blip2Base, disabled_train
from video_llama.models.video_llama import VideoLLAMA
from video_llama.common.registry import registry
from video_llama.models.ImageBind.models.imagebind_model import ImageBindModel,ModalityType
from video_llama.models.ImageBind.models import imagebind_model
from video_llama.datasets.data_utils import neg_sampler

@registry.register_model("q_former_aligned")
class QformerAligned(Blip2Base):
    def __init__(self,
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            freeze_qformer=False, # original setting is True
            num_query_token=32,
            img_size=224,
            drop_path_rate=0,
            freeze_vit=True,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            equip_audio_branch = True,
            imagebind_ckpt_path = '/mnt/workspace/ckpt'):
        super().__init__()

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

        if equip_audio_branch:
            print (f'Initializing audio encoder from {imagebind_ckpt_path} ...')
            self.audio_encoder,self.audio_hidden_size = \
                imagebind_model.imagebind_huge()
            self.audio_encoder.load_state_dict(torch.load("{}/imagebind_huge.pth".format(imagebind_ckpt_path)))
            # free vision encoder
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            print ('audio encoder initialized.')
        

        self.linear_proj = nn.Linear(self.query_tokens.size()[-1], 1024)
        self.triplet_loss_margin = 1.0
        self.cos = nn.CosineSimilarity()
        self.triplet_loss = nn.TripletMarginLoss(margin=self.triplet_loss_margin, eps=1e-7)
        

    def forward(self, images):
        # image = positive_images = samples['image']
        # assume using cc_sbu dataset
        batch_size = images.size()[0]
        negative_images = neg_sampler(batch_size=batch_size, image=images)

        q_former_branch_outputs = self.qformer_branch(image=images)
        neg_embeds = self.qformer_branch(image=negative_images)
        anchors = self.imagebind_branch(image=images)

        # calculate loss
        pos_embedding = self.select_embed(embeds=q_former_branch_outputs, anchors=anchors)
        neg_embedding = self.select_embed(embeds=neg_embeds, anchors=anchors)
        loss = self.triplet_loss(anchors, pos_embedding, neg_embedding)

        return {
            "q_former_branch_outputs": q_former_branch_outputs,
            "loss": loss 
        }

    def qformer_branch(self, image):
        device = image.device
        image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            )
        q_hidden_state = query_output.last_hidden_state
        q_hidden_state_transformed = self.linear_proj(q_hidden_state)
        return q_hidden_state_transformed

    def imagebind_branch(self, image):
        _, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(image, modality_type=ModalityType.VISION)
        return audio_imagebind_finalout

    def select_embed(self, embeds, anchors):
        num_query = embeds.size()[1]
        batch_size = embeds.size()[0]

        embed_selected_list = []
        for k in batch_size:
            embeds_list = []
            for i in num_query:
                embeds_list.append(embeds[k, i, :])

            sim_list = []
            for pos in embeds_list:
                sim_list.append(self.cos(anchors[k, :], pos)) # max similarity for positive images & hard negative images
            idx = sim_list.index(max(sim_list))

            embed_selected = embeds_list[idx]
            embed_selected_list.append(embed_selected)
        embeds_final = torch.stack(embed_selected_list, dim=0)
        return embeds_final
    
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        freeze_qformer = cfg.get("freeze_qformer", True)
        freeze_vit = cfg.get("freeze_vit", True)
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        equip_audio_branch = cfg.get("equip_audio_branch", True)
        imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", '/mnt/workspace/ckpt')

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            num_query_token=num_query_token,
            freeze_qformer=freeze_qformer,
            freeze_vit=freeze_vit,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            equip_audio_branch=equip_audio_branch,
            imagebind_ckpt_path=imagebind_ckpt_path
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)