from torch.utils.data import Dataset
from scipy import ndimage
import numpy as np
import cv2
from PIL import Image

# ---train&val data---
class SingleInputDataset(Dataset):
    def __init__(self, data_path, txt_path, train_transform=None, valid_transform=None):
        txt_path = data_path + '/' +txt_path
        lines = open(txt_path, 'r')
        imgs = []
        for line in lines:
            line = line.rstrip()
            words = line.split()
            # words = line.split(',')
            imgs.append((words[0], int(words[1])))
            

        self.imgs = imgs        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.data_path = data_path

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        if self.train_transform is not None:
            img_path = self.data_path + '/' +  img_path
        if self.valid_transform is not None:
            img_path = self.data_path + '/' + img_path
        
        img = Image.open(img_path).convert('RGB')
        
        
        if self.train_transform is not None:
            img = self.train_transform(img)
        if self.valid_transform is not None:
            img = self.valid_transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


# ---test data---
class TestDataset(Dataset):
    def __init__(self, data_path, txt_path, test_transform=None):
        txt_path = data_path + '/' + txt_path
        lines = open(txt_path, 'r')
        self.data_path = data_path
        imgs = []
        for line in lines:
            line = line.rstrip()
            words = line.split()
            # words = line.split(',')
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs        # generate the global list
        self.test_transform = test_transform#调用transform.py文件中的Data_Transforms函数

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        # img_path = "/home/ghy/shujuji/phase1/valset/" + img_path
        # img_path = self.data_path + '/' +  img_path
        img = Image.open(img_path).convert('RGB')
        
        # transform
        if self.test_transform is not None:
            img = self.test_transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
