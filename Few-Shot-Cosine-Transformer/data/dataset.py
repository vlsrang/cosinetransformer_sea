# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import pdb
import torch
from PIL import Image
import json
import numpy as np
import random
import torchvision.transforms as transforms
import os
import cv2 as cv

identity = lambda x:x

class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        
        sub_data_loader_params = dict(batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,  # use main thread only or may receive multiple batches
                                  pin_memory=False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(
                self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        batch = next(iter(self.sub_dataloader[i]))
        #print(f"Batch {i} shape: {[item.size() for item in batch]}")
        return batch

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, i):
        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed)
        
        # 이미지 로드
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        # 라벨 데이터 확인
        target = self.target_transform(self.cl)
        #print(f"Target label: {target}, Type: {type(target)}")  # 디버깅 추가
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            # 랜덤으로 n_way 클래스 샘플링
            sampled_classes = torch.randperm(self.n_classes)[:self.n_way]
            #print(f"Episode {i}: Sampled classes {sampled_classes}")
            yield sampled_classes
