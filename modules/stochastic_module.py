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

class LinearCosRadius(nn.Module):
    '''
    Define the radius (R) as the linear function of the cos-similarity (t, v)
    '''
    def __init__(self, config: Config):
        super(LinearCosRadius, self).__init__()
        self.num_frames = config.num_frames
        self.embed_dim = config.embed_dim

        self.linear_proj = nn.Linear(self.num_frames, self.embed_dim)
        self.learnable_scalar = nn.Parameter(torch.Tensor(1))
        self.run_diffusion = run_diffusion()
        self.linear_text = nn.Linear(self.embed_dim , 16)

        self.linear_sims = nn.Linear(16 , self.embed_dim)
        self.linear_sims_sum = nn.Linear(64 , 1)

        self._init_parameters()
        self.config = config


    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, video_embeds , diff_data , stage):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """

        video_embeds = self.linear_text( video_embeds )

        # run in  Diffusion-LM/improved-diffusion/scripts/diffusion_lm.py
        if stage == "Train":
            sims_out , target_loss = self.run_diffusion( video_embeds , diff_data , stage )
        if stage == "Test":
            sims_out = self.run_diffusion( video_embeds , diff_data , stage )
        
        sims_out = self.linear_sims( sims_out )  #32*64*512
        sims_out = sims_out.permute(0,2,1)   #32*512*64
        sims_out = self.linear_sims_sum( sims_out ).squeeze(2)   #32*512

        if stage == "Train":
            return sims_out , target_loss
        

        return sims_out


class StochasticText(nn.Module):
    def __init__(self, config: Config):
        super(StochasticText, self).__init__()

        self.config = config

        self.std_branch = LinearCosRadius(config)

    def forward(self, text_features, video_features , diff_data ,stage):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_texts x embed_dim
        """
        # @WJM: re-parameterization for text (independent of the text pool video)
        text_mean = text_features

        # radius
        if stage == "Train":
            log_var , tagrget_loss = self.std_branch(text_features, video_features , diff_data , stage)
        else:
            log_var = self.std_branch(text_features, video_features , diff_data , stage)
        text_std = torch.exp(log_var) # log var -> var

        # # randomness
        # if self.config.stochastic_prior == 'uniform01':
        #     sigma = torch.rand_like(text_features)

        sigma = 1

        # re-parameterization
        text_features = text_mean + sigma * text_std

        if stage == "Train":
            return text_features, text_mean, log_var , tagrget_loss
        return text_features, text_mean, log_var


