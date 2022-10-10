from .init_dataset import ImageListDataset
import torch.utils.data
import os
import pandas as pd

def generate_loader(opt, split, inference_list = None):   #载入数据集
    
    if split == 'train':
        current_transform = opt.train_transform
        current_shuffle = True
        sampler = None
        drop_last = False
        data_list = 'data\\train.txt'   #训练图像路径，放在ChaLearn_liveness_challenge-master文件夹下
        
    elif split == 'test':
        current_transform = opt.test_transform
        current_shuffle = False
        sampler = None
        drop_last = False
        data_list = 'data\\test.txt'   #训练图像路径，放在ChaLearn_liveness_challenge-master文件夹下
    elif split == 'val':
        current_transform = opt.test_transform
        current_shuffle = False
        sampler = None
        drop_last = False
        data_list = 'data\\val.txt'        
        
  
    if inference_list:
        data_list = inference_list
        data_root= '/path/to/val/and/test/data'
        current_shuffle = False
    else:
        data_root = opt.data_root   #data_root='D:\\lcy\\feathernet\\dataset\\'
        
    dataset = ImageListDataset(data_root = data_root,  data_list = data_list, transform=current_transform)

    assert dataset
    if split == 'train' and opt.fake_class_weight != 1:
        weights = [opt.fake_class_weight if x != 1 else 1.0 for x in dataset.df.label.values]
        num_samples = len(dataset)
        replacement = True
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement)
        current_shuffle = False
    if split == 'train' and len(dataset) % (opt.batch_size // opt.ngpu) < 16:   #原来批大小是128，比值是32；改为批大小32，比值是16
        drop_last = True

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = current_shuffle,
                                                 num_workers = int(opt.nthreads),sampler = sampler, pin_memory=True,
                                                 drop_last = drop_last)
    return dataset_loader
