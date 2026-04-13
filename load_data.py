from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import math

import pickle

from model.simple_tokenizer import SimpleTokenizer
import os
import numpy as np
import scipy.io as scio

from torch.utils.data import Dataset
import torch
import random
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from utils import get_args


class BaseDataset(Dataset):
    def __init__(self,
                 captions: dict,
                 indexs: dict,
                 labels: dict,
                 m1: dict,
                 m2: dict,
                 is_train=True,
                 tokenizer=SimpleTokenizer(),
                 maxWords=32,
                 imageResolution=224,
                 ):

        self.captions = captions
        self.indexs = indexs
        self.labels = labels
        self.m1 = torch.from_numpy(m1)
        self.m2 = torch.from_numpy(m2)
        self.maxWords = maxWords
        self.tokenizer = tokenizer

        self.transform = Compose([
            Resize(imageResolution, interpolation=Image.BICUBIC),
            CenterCrop(imageResolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) if is_train else Compose([
            Resize((imageResolution, imageResolution), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.__length = len(self.indexs)

    def __len__(self):
        return self.__length

    def _load_image(self, index: int) -> torch.Tensor:
        image_path = self.indexs[index].strip()
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def _load_text(self, index: int):
        captions = self.captions[index]

        if self.tokenizer is not None:
            # use_cap = captions[random.randint(0, len(captions) - 1)]
            use_cap = ''.join(captions)
            words = self.tokenizer.tokenize(use_cap)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.maxWords - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]

            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            caption = self.tokenizer.convert_tokens_to_ids(words)

            while len(caption) < self.maxWords:
                caption.append(0)

        caption = torch.tensor(caption)
        key_padding_mask = (caption == 0)
        return caption, key_padding_mask

    def _load_label(self, index: int) -> torch.Tensor:
        label = self.labels[index]
        label = torch.from_numpy(label)
        return label

    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.int64)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels

    def __getitem__(self, index):
        image = self._load_image(index)
        caption, key_padding_mask = self._load_text(index)
        label = self._load_label(index)
        m1 = self.m1[index]
        m2 = self.m2[index]
        return image, caption, key_padding_mask, label, m1, m2, index


def split_data(captions, indexs, labels, query_num, train_num, full_ratio, oimg_ratio, seed=None):

    np.random.seed(seed=1)

    random_index = np.random.permutation(range(len(indexs)))
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    # ############ 添加缺失掩码 #############
    # 训练集
    t_full_size = math.ceil(train_num * full_ratio)  # 完备数据量
    t_oimg_size = math.floor(train_num * oimg_ratio)  # 仅图像模态数据量
    t_otxt_size = train_num - t_full_size - t_oimg_size  # 仅文本模态数据量

    t_index_m = np.arange(t_full_size + t_oimg_size + t_otxt_size)
    t_m1 = np.expand_dims((t_index_m < (t_full_size + t_oimg_size)).astype(np.float32), axis=1)
    t_m2 = np.expand_dims((t_index_m < t_full_size).astype(int)
                        + (t_index_m >= (t_full_size + t_oimg_size)).astype(int), axis=1)
    np.random.shuffle(t_index_m)
    t_m1 = t_m1[t_index_m]
    t_m2 = t_m2[t_index_m]
    # 查询集
    q_m1 = np.ones((query_num, 1)).astype(np.float32)
    q_m2 = np.ones((query_num, 1)).astype(int)

    # 检索集
    db_num = retrieval_index.shape[0]
    d_full_size = math.ceil(db_num * full_ratio)  # 完备数据量
    d_oimg_size = math.floor(db_num * oimg_ratio)  # 仅图像模态数据量
    d_otxt_size = db_num - d_full_size - d_oimg_size  # 仅文本模态数据量

    d_index_m = np.arange(d_full_size + d_oimg_size + d_otxt_size)
    d_m1 = np.expand_dims((d_index_m < (d_full_size + d_oimg_size)).astype(np.float32), axis=1)
    d_m2 = np.expand_dims((d_index_m < d_full_size).astype(int)
                        + (d_index_m >= (d_full_size + d_oimg_size)).astype(int), axis=1)
    np.random.shuffle(d_index_m)
    d_m1 = d_m1[d_index_m]
    d_m2 = d_m2[d_index_m]
    # #####################################

    query_indexs = indexs[query_index]
    query_captions = captions[query_index]
    query_labels = labels[query_index]

    train_indexs = indexs[train_index]
    train_captions = captions[train_index]
    train_labels = labels[train_index]

    retrieval_indexs = indexs[retrieval_index]
    retrieval_captions = captions[retrieval_index]
    retrieval_labels = labels[retrieval_index]

    split_indexs = (query_indexs, train_indexs, retrieval_indexs)
    split_captions = (query_captions, train_captions, retrieval_captions)
    split_labels = (query_labels, train_labels, retrieval_labels)
    split_m1 = {'q_m1': q_m1, 't_m1': t_m1, 'd_m1': d_m1}
    split_m2 = {'q_m2': q_m2, 't_m2': t_m2, 'd_m2': d_m2}

    return split_indexs, split_captions, split_labels, split_m1, split_m2


def generate_dataset(captionFile: str,
                     indexFile: str,
                     labelFile: str,
                     maxWords=32,
                     imageResolution=224,
                     dataset='mirflickr',
                     query_num=2000,
                     train_num=10000,
                     full_ratio=0.1,
                     oimg_ratio=0.45,
                     seed=None,
                     ):

    if dataset == 'mirflickr':
        captions = scio.loadmat(captionFile)["caption"].squeeze(0).squeeze(-1)
        indexs = scio.loadmat(indexFile)["index"]
        labels = scio.loadmat(labelFile)["category"]
    elif dataset == 'nuswide':
        with open(captionFile, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
        # 将列表转换为 NumPy 数组
        captions = np.array(lines, dtype=str)
        indexs = scio.loadmat(indexFile)["index"]
        labels = scio.loadmat(labelFile)["category"]
    elif dataset == 'coco':
        captions = scio.loadmat(captionFile)["caption"].squeeze(0)
        indexs = scio.loadmat(indexFile)["index"]
        labels = scio.loadmat(labelFile)["category"]

    split_indexs, split_captions, split_labels, split_m1, split_m2 = \
        split_data(captions, indexs, labels, query_num=query_num, train_num=train_num,
                   full_ratio=full_ratio, oimg_ratio=oimg_ratio, seed=seed)

    query_data = BaseDataset(captions=split_captions[0], indexs=split_indexs[0], labels=split_labels[0],
                             m1=split_m1['q_m1'], m2=split_m2['q_m2'],
                             maxWords=maxWords, imageResolution=imageResolution, is_train=False)
    train_data = BaseDataset(captions=split_captions[1], indexs=split_indexs[1], labels=split_labels[1],
                             m1=split_m1['t_m1'], m2=split_m2['t_m2'],
                             maxWords=maxWords, imageResolution=imageResolution)
    retrieval_data = BaseDataset(captions=split_captions[2], indexs=split_indexs[2], labels=split_labels[2],
                                 m1=split_m1['d_m1'], m2=split_m2['d_m2'],
                                 maxWords=maxWords, imageResolution=imageResolution, is_train=False)

    return train_data, query_data, retrieval_data
