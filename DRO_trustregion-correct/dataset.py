import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import glob
import numpy as np
# from torchvision.io import read_image
import PIL.Image as Image
from torchvision.transforms.functional import normalize
from torchvision.transforms.transforms import RandomRotation, ToTensor

class AgeDataset(Dataset):
    def __init__(self,path,train=False):
        self.path = path
        self.num_imgs = len(glob.glob(path+'/*/*/*'))
        # print(self.num_imgs)
        self.img_list = glob.glob(path+'/*/*/*')
        # print(self.img_list)
        if train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0,1),
                # transforms.ToTensor(),
                transforms.RandomCrop(60,4,pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(0,1),
                transforms.Resize(248),
            ])

    def __len__(self):
        # print('__len__ of dataset = ', self.num_imgs)
        return self.num_imgs

    def __getitem__(self,idx):
        if self.img_list[idx].endswith('jpg'):
            idx=idx
        else:
            idx=idx+1
        img = Image.open(self.img_list[idx])
        temp_list = self.img_list[idx].split('/')
        # age = int(temp_list[-2])
        age = int(temp_list[-3])
        # label = torch.zeros(75-15+1,2)
        # # label = torch.zeros(72-15,2)
        # label[:age-15] = torch.tensor([1,0])
        # label[age-15:] = torch.tensor([0,1])
        ## 变成一维试试看
        label=torch.tensor(age/75).float()
        # img = (transforms.ToTensor()(img)-0.5)*2
        img = self.transform(img)
        return img,label,age