import gc
import time
import torch
import numpy as np
from tqdm import tqdm
from config.base_config import Config
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id_diff


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0


    def _train_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):
            data['diff-text'] = data['text']
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
            
            data['video'] = data['video'].to(self.device)

            _ , _ , video_embeds_pooled, text_embeds_stochastic, target_loss = self.model(data, is_train=True , return_all_frames=True)

            output = sim_matrix_training(text_embeds_stochastic, video_embeds_pooled, self.pooling_type)
            loss = self.loss(output, self.model.clip.logit_scale)

            loss_all =  loss + target_loss * 0.1
            loss_all.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1

            total_loss += loss_all.detach().item()


            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Total Loss: {:.6f}, Original Loss: {:.6f}, Diffusion Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss_all.detach().item(),
                    loss.detach().item(),
                    target_loss.detach().item(),
                    ))


            if batch_idx in eval_steps:

                    val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                    self.model.train()

                    if val_res['R1-window'] > self.best_window:
                        self.best_window = val_res['R1-window']
                        self._save_checkpoint(epoch, save_best=True)

                    if val_res['R1'] > self.best:
                        self.best = val_res['R1']

                    print(" Current Best Window Average R@1 is {}".format(self.best_window))
                    print(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    
    def _valid_epoch_step(self, epoch, step, num_steps):

        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        all_row_text = []

        with torch.no_grad():
            print( len(self.valid_data_loader) )
            for idx, data in tqdm(enumerate(self.valid_data_loader)):
                data['diff-text'] = data['text']
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                data['video'] = data['video'].to(self.device)

                text_embed, vid_embed, _, _ = self.model(data, return_all_frames=True, is_train=False)

                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())
                for raw_data in data['diff-text']: 
                    all_row_text.append( raw_data)

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id) 

            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)

            vid_embeds_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]

            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

            self.model.pool_frames.cpu()
            vid_embeds_pooled = self.model.pool_frames(text_embeds, vid_embeds)
            self.model.pool_frames.cuda()

            self.model.textdiffs.cpu()

            text_embeds_diff_allpairs = torch.zeros(size=(vid_embeds.shape[0], text_embeds.shape[0], text_embeds.shape[1]))

            vid_embeds = vid_embeds.cuda()
            text_embeds_diff_allpairs = text_embeds_diff_allpairs.cpu()
            text_embeds = text_embeds.cuda()
            self.model.textdiffs.cuda()
            for (idx_vid, single_vid) in tqdm((enumerate(vid_embeds))):

                single_vid_vec = single_vid.unsqueeze(0)
                single_vid_repeat = single_vid_vec.tile((text_embeds.shape[0], 1, 1))
                all_text_embed, _ = self.model.textdiffs(text_embeds, single_vid_repeat , all_row_text,'Test')
                all_text_embed = all_text_embed.cpu()
                text_embeds_diff_allpairs[idx_vid,:,:] = all_text_embed
            
            vid_embeds_pooled = vid_embeds_pooled.cpu()
            text_embeds_diff_allpairs = text_embeds_diff_allpairs.cpu()
            self.model.textdiffs.cpu()
            self.model.textdiffs.cuda()

            text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id_diff(text_embeds_diff_allpairs,
                    vid_embeds_pooled, all_vid_ids, self.pooling_type)


            sims = sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)

            total_val_loss = total_val_loss / len(self.valid_data_loader)


            metrics = self.metrics
            res = metrics(sims)

            for m in res:
                self.window_metric[m].append(res[m])

            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")

            res['loss_val'] =  total_val_loss

            return res
