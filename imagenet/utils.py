from argparse import ArgumentParser
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_folder', help="Input file folder with images")

    args = parser.parse_args()
    return args.data_folder

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f)
    return dict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ImageNetDS(Dataset):
    """ImageNet downsampled dataset."""
    def __init__(self, datafile, img_size, idx, mean_img=None, 
                train=False, fp16=False):
        super(ImageNetDS, self).__init__()
        self.img_size = img_size
        self.fp16 = fp16
        if train:
            self.img_dict = unpickle(datafile + str(idx))
            self.mean_img = self.img_dict['mean']
            self.mean_img = self.mean_img / np.float32(255)
        else:
            self.img_dict = unpickle(datafile)
            self.mean_img = mean_img
        self.image = self.img_dict['data']
        self.label = self.img_dict['labels']
        self.transform()

    def transform(self):
        # image, label = sample['image'], sample['label']
        # normalize
        self.image = self.image / np.float32(255)
        self.label = np.array([i-1 for i in self.label])
        # remove mean from images, computed from training data
        self.image -= self.mean_img

        if self.img_size == '32x32':
            self.image = np.dstack((self.image[:,:1024], self.image[:, 1024:2048], self.image[:, 2048:]))
            self.image = self.image.reshape((self.image.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)
        elif self.img_size =='16x16':
            self.image = np.dstack((self.image[:,:256], self.image[:, 256:512], self.image[:, 512:]))
            self.image = self.image.reshape((self.image.shape[0], 16, 16, 3)).transpose(0, 3, 1, 2)
        else:
            raise NotImplementedError
        self.image = torch.from_numpy(self.image)
        self.label = torch.from_numpy(self.label)

    def __len__(self):
        return len(self.img_dict['data'])

    def __getitem__(self, idx):
        return {'image': self.image[idx], 'label': self.label[idx]}


def load_mean(data_folder, fp16=False):
    datafile = os.path.join(data_folder, 'train_data_batch_1')
    d = unpickle(datafile)
    mean = d['mean']
    mean = mean / np.float32(255)
    return mean

if __name__ == '__main__':
    data_folder = parse_arguments()
    val_data = os.path.join(data_folder, 'val_data')
    train_datafile = os.path.join(data_folder, 'train_data_batch_')
    mean_img = load_mean(data_folder)

    train_dataset = ImageNetDS(train_datafile, '16x16', 1, mean_img=None, train=True)
    train_loader = DataLoader(train_dataset, batch_size=256,
                            shuffle=True, num_workers=8)

    val_dataset = ImageNetDS(val_data, '16x16', 1, mean_img=mean_img, train=False)
    val_loader = DataLoader(val_dataset, batch_size=256,
                                shuffle=False, num_workers=4)

    for i, sample_batched in enumerate(train_loader):
        print('TRAIN',i, sample_batched['image'])
        if i > 0: break

    for i, sample_batched in enumerate(val_loader):
        print('VAL', i, sample_batched['image'])
        if i > 0: break

    # print('mini-batch number: ', len(train_loader))
