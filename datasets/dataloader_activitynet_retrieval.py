from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
from torch.utils.data import Dataset
from config.base_config import Config
import numpy as np
import json
import math
from datasets.rawvideo_util import RawVideoExtractor
from datasets.tokenization_clip import SimpleTokenizer as ClipTokenizer

class ActivityNet_DataLoader(Dataset):
    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.data_path = "data/ActivityNet"
        self.img_transforms = img_transforms
        self.features_path = config.videos_dir
        self.feature_framerate = 1
        self.num_frames = config.num_frames
        feature_framerate = self.feature_framerate
        self.max_words = 64
        self.max_frames = 64
        image_resolution = 224
        self.tokenizer =  ClipTokenizer()
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = 0
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = 2
        assert self.slice_framepos in [0, 1, 2]

        self.subset = split_type
        if split_type != "train":
            self.subset = "val"
        assert self.subset in ["train", "val"]

        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_ids.json")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_ids.json")

        video_json_path_dict = {}
        video_json_path_dict["train"] = os.path.join(self.data_path, "train.json")
        video_json_path_dict["val"] = os.path.join(self.data_path, "val_1.json")

        pseudo_video_id_list, video_id_list = self._get_video_id_single(video_id_path_dict[self.subset])
        pseudo_caption_dict = self._get_captions_single(video_json_path_dict[self.subset])

        print("pseudo_video id list: {}".format(len( pseudo_video_id_list)))
        print("video id list: {}".format(len(video_id_list)))
        print("pseudo caption dict: {}".format(len(pseudo_caption_dict.keys())))

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_id_list:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict
        print("video dict: {}".format(len(video_dict)))

        self.pseudo_video_id_list = pseudo_video_id_list
        self.video_id_list = video_id_list
        self.pseudo_caption_dict = pseudo_caption_dict

        # Get iterator video ids
        self.video_id2idx_dict = {pseudo_video_id: id for id, pseudo_video_id in enumerate(self.pseudo_video_id_list)}
        # Get all captions
        self.iter2video_pairs_dict = {}
        for pseudo_video_id, video_id in zip(self.pseudo_video_id_list, self.video_id_list):
            if pseudo_video_id not in self.pseudo_caption_dict or video_id not in self.video_dict:
                continue
            caption = self.pseudo_caption_dict[pseudo_video_id]
            n_caption = len(caption['start'])
            for sub_id in range(n_caption):
                self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (pseudo_video_id, sub_id)

        self.rawVideoExtractor = RawVideoExtractor(framerate=1, size=224)

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    def _get_video_id_from_pseduo(self, pseudo_video_id):
        video_id = pseudo_video_id[:]
        return video_id

    def _get_video_id_single(self, path):
        pseudo_video_id_list = []
        video_id_list = []
        print('Loading json: {}'.format(path))
        with open(path, 'r') as f:
            json_data = json.load(f)

        for pseudo_video_id in json_data:
            if pseudo_video_id in pseudo_video_id_list:
                print("reduplicate.")
            else:
                video_id = self._get_video_id_from_pseduo(pseudo_video_id)
                pseudo_video_id_list.append(pseudo_video_id)
                video_id_list.append(video_id)
        return pseudo_video_id_list, video_id_list

    def _get_captions_single(self, path):
        pseudo_caption_dict = {}
        with open(path, 'r') as f:
            json_data = json.load(f)

        for pseudo_video_id, v_ in json_data.items():
            pseudo_caption_dict[pseudo_video_id] = {}
            duration = v_["duration"]
            pseudo_caption_dict[pseudo_video_id]["start"] = np.array([0], dtype=object)
            pseudo_caption_dict[pseudo_video_id]["end"] = np.array([int(math.ceil(float(duration)))], dtype=object)
            pseudo_caption_dict[pseudo_video_id]["text"] = np.array([" ".join(v_["sentences"])], dtype=object)
        return pseudo_caption_dict

    def _get_text(self, pseudo_video_id, sub_id):
        caption = self.pseudo_caption_dict[pseudo_video_id]
        k = 1
        r_ind = [sub_id]

        starts = np.zeros(k, dtype=np.int32)
        ends = np.zeros(k, dtype=np.int32)
        words = ""

        for i in range(k):
            ind = r_ind[i]
            start_, end_ = caption['start'][ind], caption['end'][ind]
            starts[i], ends[i] = start_, end_
            words = (caption['text'][ind])

        return words, starts, ends

    def _get_rawvideo(self, idx, s, e):
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.int32)
        max_video_length = [0] * len(s)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(s), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)
        video_path = self.video_dict[idx]

        for i in range(len(s)):
                start_time = int(s[i])
                end_time = int(e[i])
                start_time = start_time if start_time >= 0. else 0.
                end_time = end_time if end_time >= 0. else 0.
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = end_time + 1

                # Should be optimized by gathering all asking of this video
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time,
                                                     sample_type='uniform', num_frames=self.num_frames)
                
        return raw_video_data

    def __getitem__(self, feature_idx):
        pseudo_video_id, sub_id = self.iter2video_pairs_dict[feature_idx]
        idx = self.video_id2idx_dict[pseudo_video_id]

        caption, starts, ends = self._get_text(pseudo_video_id, sub_id)
        # print( starts )
        video = self._get_rawvideo(self.video_id_list[idx], starts, ends)

        if self.img_transforms is not None:
            imgs = self.img_transforms(video)

        return {
            'video_id': (pseudo_video_id),  #str
            'video': (imgs),     #numpy
            'text': (caption),   #dict
        }
