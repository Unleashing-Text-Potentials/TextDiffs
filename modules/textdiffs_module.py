from pathlib import Path
current_file = Path(__file__)
current_folder = current_file.parent.parent
import os
current_folder = str(current_folder).lstrip('/')
current_folder = os.path.join( (current_folder) ,  'Diffusion-LM', 'improved-diffusion')
import sys
sys.path.append( "/" + str(current_folder))

import torch
import torch.nn as nn
from config.base_config import Config
from scripts.diffusion_lm import run_diffusion

class Diffusion(nn.Module):
    def __init__(self, config: Config):
        super(Diffusion, self).__init__()
        self.num_frames = config.num_frames
        self.embed_dim = config.embed_dim

        self.diff_proj = nn.Linear(self.num_frames, self.embed_dim)
        self.run_diffusion = run_diffusion()

        self.diff_text_dim = 16
        self.diff_text_patch = 64
        self.diff_text_cls = 1
        self.linear_text = nn.Linear(self.embed_dim , self.diff_text_dim)

        self.linear_sims = nn.Linear(self.diff_text_dim , self.embed_dim)
        self.linear_sims_sum = nn.Linear(self.diff_text_patch , self.diff_text_cls)

        self._init_parameters()
        self.config = config


    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, video_embeds , diff_data , stage):

        video_embeds = self.linear_text( video_embeds )

        # run in  Diffusion-LM/improved-diffusion/scripts/diffusion_lm.py
        if stage == "Train":
            sims_out , target_loss = self.run_diffusion( video_embeds , diff_data , stage )
        if stage == "Test":
            sims_out = self.run_diffusion( video_embeds , diff_data , stage )
        
        sims_out = self.linear_sims( sims_out )  #batch x diff_text x embed_dim
        sims_out = sims_out.permute(0,2,1)   #batch x embed_dim x diff_text
        sims_out = self.linear_sims_sum( sims_out ).squeeze(2)    #batch x embed_dim

        if stage == "Train":
            return sims_out , target_loss
        

        return sims_out


class TextDiffs(nn.Module):
    def __init__(self, config: Config):
        super(TextDiffs, self).__init__()

        self.config = config

        self.diffusion = Diffusion(config)

    def forward(self, text_features, video_features , diff_data ,stage):

        if stage == "Train":
            l_text_diff , tagrget_loss = self.diffusion(video_features , diff_data , stage)
        else:
            l_text_diff = self.diffusion(video_features , diff_data , stage)
        text_diff = torch.exp(l_text_diff)

        text_features = text_features +  text_diff

        if stage == "Train":
            return text_features,  tagrget_loss
        return text_features , l_text_diff


