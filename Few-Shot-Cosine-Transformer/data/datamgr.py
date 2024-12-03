# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import pdb
import torch
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SetDataset, EpisodicBatchSampler
from abc import abstractmethod

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type == 'Resize':
            return method(self.image_size)
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        # elif transform_type=='Scale':
        #     return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
                transform_list = ['Resize','RandomResizedCrop', 'ColorJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']
            
        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 





def seed_worker(worker_id):
    worker_seed = worker_id #torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, k_shot, n_query, n_episode):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = k_shot + n_query
        self.n_episode = n_episode
        self.trans_loader = TransformLoader(image_size)
    

    


    # parameters that would change on train/val set
    def get_data_loader(self, data_file, aug):
        g = torch.Generator()
        g.manual_seed(0)
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(data_file, self.batch_size,
                             transform)
        sampler = EpisodicBatchSampler(
            len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler = sampler, num_workers = 8, pin_memory = True, collate_fn=custom_collate_fn, worker_init_fn=seed_worker, generator=g)     
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

def custom_collate_fn(batch):
    images, labels = zip(*batch)  # 이미지와 라벨 분리

    # 이미지 병합
    max_size = max([img.size(0) for img in images])
    padded_images = []
    for img in images:
        padding = torch.zeros((max_size - img.size(0), *img.size()[1:]))
        padded_images.append(torch.cat([img, padding], dim=0))

    # 라벨 병합 (패딩)
    max_label_size = max([len(label) for label in labels])
    padded_labels = []
    for label in labels:
        padding = torch.full((max_label_size - len(label),), -1, dtype=torch.long)  # 패딩 값으로 -1 사용
        padded_labels.append(torch.cat([torch.tensor(label, dtype=torch.long), padding]))

    return torch.stack(padded_images), torch.stack(padded_labels)



