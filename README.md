# Predictive Completion Enhanced Deep Hashing with Auxiliary Code Guidance for Incomplete Cross-Modal Retrieval


## 1. Introduction
This is the source code of paper "Predictive Completion Enhanced Deep Hashing with Auxiliary Code Guidance for Incomplete Cross-Modal Retrieval".


## 2. Requirements
- python 3.8.0
- pytorch 2.4.1
- torchvision 0.19.1
- numpy
- scipy
- tqdm
- pillow
- einops
- ftfy
- ...


## 3. Preparation

### 3.1 Download CLIP pretrained model

Pretrained CLIP model could be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32". 

You should copy ViT-B-32.pt to this dir.

### 3.2 Generate dataset

### Processing dataset
Before training, you need to download the oringal data from [coco](https://cocodataset.org/#download)(include 2017 train,val and annotations), [nuswide](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)(include all), [mirflickr25k](https://www.kaggle.com/datasets/paulrohan2020/mirflickr25k)(include mirflickr25k and mirflickr25k_annotations_v080), 
then use the "data/make_XXX.py" to generate .mat file

After all mat file generated, the dir of `dataset` will like this:
~~~
dataset
├── base.py
├── __init__.py
├── dataloader.py
├── coco
│   ├── caption.mat 
│   ├── index.mat
│   └── label.mat 
├── flickr25k
│   ├── caption.mat
│   ├── index.mat
│   └── label.mat
└── nuswide
    ├── caption.txt  # Notice! It is a txt file!
    ├── index.mat 
    └── label.mat
~~~
> (Source: [DCHMT](https://github.com/kalenforn/DCHMT))


## 4. Train

> python main.py
