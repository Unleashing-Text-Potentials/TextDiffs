# TextDiffs:  Text Potentials for Text-Video Retrieval 

---

> **Abstract:** *Despite the recent surge in advancements in text-video retrieval technology, the full potential of textual data remains unleashed. Pre-trained CLIP-based methods have been prevailing for years but suffer from the inherent information imbalance between text and video. Existing works attempt strategies to expand text semantics, but these strategies remain confined to the text domain, limiting the extent of semantic expansion and disconnecting them from the visual domain, making it difficult to bridge the natural modality gap. In this paper, we propose a visual-guided text embedding modeling method based on continuous diffusion, TextDiff, which leverages the diffusion process to seamlessly transfer text semantics into the visual space. Specifically, we follow the diffusion process, using the distance between visual and text embeddings as a condition to diffuse text embeddings toward the visual direction. We design a dual supervision training method with a teacher-student loss and a retrieval loss to provide enhanced supervisory signals for text embedding diffusion. Extensive experiments on five commonly used text-video retrieval benchmarks (including MSRVTT, LSMDC, MSVD, ActivityNet Captions, and DiDeMo) demonstrate the effectiveness of our approach, achieving superior performance.*


><p align="center">
> <img width="940" src="">
> </p>


---

## Content

1. [Dependencies](#Dependencies)
1. [Diffusion](#Diffusion)
1. [Dataset](#Dataset)
1. [Evaluation](#Evaluation)
1. [Training](#Training)
1. [Acknowledgement](#Acknowledgement)


## Dependencies

- PyTorch 1.12.1 
- OpenCV 4.7.0
- transformers 4.30.2
- mpi4py  3.0.1

## Diffusion

Get the weights of our trained diffusion model [link](https://drive.google.com/drive/folders/1B_BsN0lxGoRb0XOA9wo-EH8ipqAVvdWg?usp=sharing) and into  `./Diffusion-LM/diffusion_models`

## Dataset

| Dataset | Video Clips | Text-Video Pairs | Link |
|:-----------:|:-----------:| :-----------: | :-----------: |
|MSR-VTT|10K|one-to-twenty|[link](https://github.com/ArrowLuo/CLIP4Clip)|
|LSMDC|118081|one-to-one|[link](https://github.com/ArrowLuo/CLIP4Clip)|
|DiDeMo|10464|one-to-many|[link](https://drive.google.com/drive/u/1/folders/1_oyJ5rQiZboipbMl6tkhY8v0s9zDkvJc)|
|ActivityNet|20K|one-to-many|[link](https://github.com/jpthu17/EMCL)|


## Evaluation

Download the checkpoints into `./outputs/{Dataset}/{FOLDER_NAME_UNDER_*Dataset*}`. 

We adopt `--seed=24` and `--diffusion_steps=10` for all methods.  One may consider specifying `--save_memory_mode` for larger datasets or computational-constrained platforms at evaluation. Same as T-MASS, the evaluation is default to text-to-video retrieval performance (i.e., `--metric=t2v`), for video-to-text retrieval performance, specify `--metric=v2t`. For post processing operation evaluation results of DSL, specify `--DSL`. 

Replace `{videos_dir}` with the path to the dataset.

| Dataset | Command | t2v R@1 Result |
|:-----------:|:-----------:| :-----------: |
|MSR-VTT-9k|`python test.py --datetime={FOLDER_NAME_UNDER_MSR-VTT-9k}   --arch=clip_stochastic   --videos_dir={VIDEO_DIR}  --batch_size=32 --noclip_lr=1e-7 --transformer_dropout=0.3  --dataset_name=MSRVTT --msrvtt_train_file=9k   --gpu='0'  --load_epoch=0   --exp_name=MSR-VTT-9k --diffusion_steps=10`| 54.3 |
|LSMDC|`python test.py --arch=clip_stochastic    --exp_name=LSMDC --videos_dir={VIDEO_DIR}  --batch_size=32 --noclip_lr=1e-7 --transformer_dropout=0.3 --dataset_name=LSMDC --gpu='0' --num_epochs=5   --load_epoch=0 --datetime={FOLDER_NAME_UNDER_LSMDC} --diffusion_steps=10`|32.13|
|DiDeMo|`python test.py  --num_frame=12 --raw_video  --arch=clip_stochastic  --exp_name=DiDeMo --videos_dir={VIDEO_DIR} --batch_size=32 --noclip_lr=1e-7 --transformer_dropout=0.4  --dataset_name=DiDeMo  --gpu='0' --num_epochs=5   --load_epoch=0 --datetime={FOLDER_NAME_UNDER_DiDeMo} --diffusion_steps=10`|52.88|
|ActivityNet|`python test.py --arch=clip_stochastic    --exp_name=ActivityNet --videos_dir={VIDEO_DIR}  --batch_size=32 --noclip_lr=1e-7 --transformer_dropout=0.3 --dataset_name=ActivityNet --gpu='0' --num_epochs=5   --load_epoch=0 --datetime={FOLDER_NAME_UNDER_LSMDC} --diffusion_steps=10`||

## Training 
Run the following training code to resume the above results. Take MSRVTT as an example, one may consider  the number of diffusion steps by specifying `--diffusion_steps. `--evals_per_epoch` can be enlarged to select a better checkpoint. The CLIP model is default to `--clip_arch=ViT-B/32`. To train on a larger CLIP backbone, speficy `--clip_arch=ViT-B/16`. 

| Dataset | Command |
|:-----------:|:-----------:|
|MSR-VTT-9k|`python train.py  --arch=clip_stochastic    --exp_name=MSR-VTT-9k --videos_dir={VIDEO_DIR}  --batch_size=32 --noclip_lr=1e-7 --transformer_dropout=0.3  --dataset_name=MSRVTT --msrvtt_train_file=9k  --gpu='0' --num_epochs=5  --diffusion_steps=10 `|
|LSMDC|`python train.py --arch=clip_stochastic   --exp_name=LSMDC --videos_dir={VIDEO_DIR}  --batch_size=32 --noclip_lr=1e-7 --transformer_dropout=0.3 --dataset_name=LSMDC   --gpu='0'  --num_epochs=5  --diffusion_steps=10`|
|DiDeMo|`python train.py  --num_frame=12 --raw_video  --arch=clip_stochastic   --exp_name=DiDeMo --videos_dir={VIDEO_DIR} --batch_size=32 --noclip_lr=1e-7 --transformer_dropout=0.4 --dataset_name=DiDeMo  --gpu='0' --num_epochs=5  --diffusion_steps=10`|
|ActivityNet|`python train.py --arch=clip_stochastic   --exp_name=ActivityNet --videos_dir={VIDEO_DIR} --batch_size=32 --noclip_lr=1e-7 --transformer_dropout=0.4 --dataset_name=ActivityNet  --gpu='0' --num_epochs=5  --diffusion_steps=10`|

## Acknowledgement

This code is built on [T-MASS](https://github.com/Jiamian-Wang/T-MASS-text-video-retrieval). Great thanks to them!
