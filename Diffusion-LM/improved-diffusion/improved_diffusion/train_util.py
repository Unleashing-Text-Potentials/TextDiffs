import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.optim import AdamW

from . import dist_util, logger
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        microbatch,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        checkpoint_path='',
        gradient_clipping=-1.,
        eval_data=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.eval_data = eval_data
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.gradient_clipping = gradient_clipping

        self.step = 0
        self.resume_step = 0

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self.checkpoint_path = checkpoint_path # DEBUG **

        if self.use_fp16:
            self._setup_fp16()

        self.use_ddp = False
        self.ddp_model = self.model


    def run_loop(self , text_data , out_dict , video_data , only_sims = False):
            batch, cond = text_data , out_dict
            video_feature = video_data
            # print( batch.size() ) 64*64*16
            return self.run_step(batch, cond , video_feature , only_sims)

    def run_step(self, batch, cond , video_feature , only_sims):
        return self.forward_backward(batch, cond , video_feature , only_sims)


    def forward_backward(self, batch, cond , video_feature , only_sims):
            micro = batch.to(dist_util.dev())
            micro_cond = {
                k: v[:].to(dist_util.dev()) for k, v in cond.items()
            }
            video_feature = video_feature.to(dist_util.dev())

            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            sims_out = self.diffusion.training_losses(
                self.ddp_model,
                micro,
                t,
                video_feature = video_feature,
                only_sims = only_sims , 
                model_kwargs=micro_cond,
            )

            return sims_out