"""
Train a diffusion model on images.
"""

import argparse
import json, torch, os
import numpy as np
import torch.nn as nn
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from transformers import AutoTokenizer
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp , denoised_fn_round
from improved_diffusion.rounding import load_models, load_tokenizer
import torch.distributed as dist
from improved_diffusion.nn import mean_flat

class run_diffusion(nn.Module):
    def __init__(self):
        super(run_diffusion, self).__init__()

        self.args, _ = create_argparser().parse_known_args()

        current_path = os.getcwd()

        "------------------------------teacher---------------------------------------------"
        self.args.checkpoint_path = "Diffusion-LM/diffusion_models"
        self.args.model_arch = "transformer"
        self.args.modality = "e2e-tgt" 
        self.ave_interval = 50000  
        self.args.diffusion_steps = 10
        self.args.noise_schedule = "sqrt"  
        self.args.use_kl = False 
        self.args.learn_sigma = False  
        self.args.image_size = 8 
        self.args.num_channels = 128 
        self.args.dropout = 0.1 
        self.args.in_channel = 16 
        self.args.out_channel = 16 
        self.args.padding_mode = "pad" 
        self.args.experiment = "random"  
        self.args.weight_decay = 0.0 
        self.args.num_res_blocks = 2  
        self.args.predict_xstart = True 
        self.args.training_mode = "e2e" 
        self.args.vocab_size = 821  
        self.args.e2e_train = "../datasets/e2e_data"

        self.teacher_model, self.teacher_diffusion = create_model_and_diffusion(
            **args_to_dict(self.args, model_and_diffusion_defaults().keys())
        )
        self.teacher_schedule_sampler = create_named_schedule_sampler(self.args.schedule_sampler, self.teacher_diffusion)

        state_dict  =torch.load("Diffusion-LM/diffusion_models/ema_0.9999_200000.pt")
        model_dict = self.teacher_model.state_dict()
        # 只更新匹配的键值对
        matched_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(matched_dict)
        self.teacher_model.load_state_dict(model_dict)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_train_loop = TrainLoop(
                model=self.teacher_model,
                diffusion=self.teacher_diffusion,
                microbatch=self.args.microbatch,
                ema_rate=self.args.ema_rate,
                log_interval=self.args.log_interval,
                save_interval=self.args.save_interval,
                resume_checkpoint=self.args.resume_checkpoint,
                use_fp16=self.args.use_fp16,
                fp16_scale_growth=self.args.fp16_scale_growth,
                schedule_sampler=self.teacher_schedule_sampler,
                weight_decay=self.args.weight_decay,
                checkpoint_path=self.args.checkpoint_path,
                gradient_clipping=self.args.gradient_clipping,
            )
        
        "---------------------------student-------------------------------------------------"

        self.args.diffusion_steps = 10
        self.student_model, self.student_diffusion = create_model_and_diffusion(
            **args_to_dict(self.args, model_and_diffusion_defaults().keys())
        )
        self.student_schedule_sampler = create_named_schedule_sampler(self.args.schedule_sampler, self.student_diffusion)
        
        self.student_train_loop = TrainLoop(
                model=self.student_model,
                diffusion=self.student_diffusion,
                microbatch=self.args.microbatch,
                ema_rate=self.args.ema_rate,
                log_interval=self.args.log_interval,
                save_interval=self.args.save_interval,
                resume_checkpoint=self.args.resume_checkpoint,
                use_fp16=self.args.use_fp16,
                fp16_scale_growth=self.args.fp16_scale_growth,
                schedule_sampler=self.student_schedule_sampler,
                weight_decay=self.args.weight_decay,
                checkpoint_path=self.args.checkpoint_path,
                gradient_clipping=self.args.gradient_clipping,
            )

    def forward(self , video_data , diff_data , stage):
            rev_tokenizer = None
            model2, tokenizer = load_models(self.args.modality, self.args.experiment, self.args.model_name_or_path, self.args.in_channel,
                                                self.args.checkpoint_path, extra_args=self.args)

            rev_tokenizer = {v: k for k, v in tokenizer.items()}
            data , out_dict = load_data_text(
                    data_dir=diff_data,
                    image_size=self.args.image_size,
                    class_cond=self.args.class_cond,
                    data_args=self.args,
                    task_mode=self.args.modality,
                    padding_mode=self.args.padding_mode,  # block, pad
                    split='valid',
                    load_vocab=rev_tokenizer,
                    model=model2,
                )
            
            if stage == "Train":

                def get_mapping_func(args, diffusion, data):
                    model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                                    args.checkpoint_path, extra_args=args)
                    model3 = get_weights(model2, args)
                    print(model3, model3.weight.requires_grad)
                    mapping_func = partial(compute_logp, args, model3.cuda())
                    diffusion.mapping_func = mapping_func
                    return mapping_func

                get_mapping_func(self.args, self.teacher_diffusion, data)
                teacher_data = self.teacher_train_loop.run_loop( text_data = data , out_dict = out_dict , video_data = video_data )

                get_mapping_func(self.args, self.student_diffusion, data)
                student_data = self.student_train_loop.run_loop( text_data = data , out_dict = out_dict , video_data = video_data )
                target_loss = mean_flat((teacher_data - student_data) ** 2)

                return student_data , target_loss.sum()
            if stage == "Test":
                sims_out = self.student_train_loop.run_loop( text_data = data , out_dict = out_dict , video_data = video_data  , only_sims = True)
                model3 = get_weights(model2, self.args)
                model_kwargs = {}
                sample_fn = self.student_diffusion.p_sample_loop if not self.args.use_ddim else self.student_diffusion.ddim_sample_loop

                sample_shape = (sims_out.shape[0], self.args.image_size ** 2, self.args.in_channel)

                sample = sample_fn(
                    self.student_model,
                    sims_out,
                    sample_shape,
                    clip_denoised=self.args.clip_denoised,
                    denoised_fn=partial(denoised_fn_round, self.args, model3.cuda()) if self.args.clamp == 'clamp' else None,
                    model_kwargs=model_kwargs,
                    top_p =self.args.top_p,
                )

                if self.args.model_arch == '1d-unet':
                    sample = sample.permute(0, 2, 1)

                return sample


def create_argparser():
        defaults = dict(
            data_dir="",
            clip_denoised=False,
            schedule_sampler="uniform",
            lr=1e-4,
            use_ddim=False,
            weight_decay=0.0,
            lr_anneal_steps=0,
            microbatch=-1,  # -1 disables microbatches
            ema_rate="0.9999",  # comma-separated list of EMA values
            log_interval=50,
            save_interval=50000,
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
            seed=101,
            gradient_clipping=-1.0,
            eval_interval=2000,
            checkpoint_path='diff_models'
        )
        text_defaults = dict(modality='text',
                            dataset_name='wikitext',
                            dataset_config_name='wikitext-2-raw-v1',
                            config='diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml',
                            model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                            experiment='gpt2_pre_compress',model_arch='conv-unet',
                            roc_train='diffusion_lm/ROCstory',#'diffusion_lm/ROCstory/ROCstory17.csv',
                            wiki_train='diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
                            e2e_train='e2e_data',
                            yelp_train='diffusion_lm/yelpnlg-resources/yelpnlg-corpus',
                            commonGen_train = 'diffusion_lm/common-gen/commongen_data',
                            emb_scale_factor=1.0, noise_level=0.0, cache_mode='no', use_bert_tokenizer='no',
                            padding_mode='block',
                            preprocessing_num_workers=1,
                            clamp='clamp',
                            top_p=-1.)
        defaults.update(model_and_diffusion_defaults())
        defaults.update(text_defaults)
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        parser.add_argument('--arch', type=str, default=None , help='First argument')
        parser.add_argument('--exp_name', type=str, default=None , help='First argument')
        parser.add_argument('--videos_dir', type=str, default=None , help='First argument')
        parser.add_argument('--noclip_lr', type=str, default=None , help='First argument')
        parser.add_argument('--transformer_dropout', type=str, default=None , help='First argument')
        parser.add_argument('--gpu', type=str, default=None , help='First argument')
        parser.add_argument('--num_epochs', type=str, default=None , help='First argument')
        parser.add_argument('--msrvtt_train_file', type=str, default=None , help='First argument')
        parser.add_argument('--datetime', type=str, default=None , help='First argument')
        parser.add_argument('--load_epoch', type=str, default=None , help='First argument')
        parser.add_argument('--num_frame', type=int, default=None , help='First argument')
        parser.add_argument('--raw_video', action='store_true', default=False, help='For Charades dataest. if specified, will load video format of .mp4')
        parser.add_argument('--save_memory_mode', action='store_true', default=False, help='t')
        parser.add_argument('--metric', type=str, default='t2v', help="'t2v'/'v2t'")
        parser.add_argument('--clip_arch', type=str, default='ViT-B/32', choices=['ViT-B/32', 'ViT-B/16'], help="CLIP arch. only when not using huggingface")
        return parser
