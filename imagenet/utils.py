from argparse import ArgumentParser
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
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

class ImageNetDS(Dataset):
    """ImageNet downsampled dataset."""
    def __init__(self, datafile, img_size, transform=None):
        super(ImageNetDS, self).__init__()
        self.datafile = datafile
        self.img_size = img_size
        self.img_dict = unpickle(self.datafile)
        self.transform = transform
        self.x = self.img_dict['data']
        self.y = self.img_dict['labels']
        self.x = self.x/np.float32(255)
        self.y = [i-1 for i in self.y]

        if self.img_size == '32x32':
            self.x = np.dstack((self.x[:,:1024], self.x[:, 1024:2048], self.x[:, 2048:]))
            self.x = self.x.reshape((self.x.shape[0], 32, 32, 3))
        elif self.img_size =='16x16':
            self.x = np.dstack((self.x[:,:256], self.x[:, 256:512], self.x[:, 512:]))
            self.x = self.x.reshape((self.x.shape[0], 16, 16, 3))
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.img_dict['data'])

    def __getitem__(self, idx):
        sample = {'image': self.x[idx],
                    'label': self.y[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_data(data_folder, img_size):
    val_data = os.path.join(data_folder, 'val_data')
    d = unpickle(val_data)
    x = d['data']
    y = d['labels']
    print('x length: ', len(x))

    if img_size == '32x32':
        x = np.dstack((x[:,:1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3))
    elif img_size =='16x16':
        x = np.dstack((x[:,:256], x[:, 256:512], x[:, 512:]))
        x = x.reshape((x.shape[0], 16, 16, 3))
    else:
        raise NotImplementedError

    return x, y

if __name__ == '__main__':
    data_folder = parse_arguments()
    val_data = os.path.join(data_folder, 'val_data')

    val_dataset = ImageNetDS(val_data, '16x16')
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        print(i, sample['image'].shape, sample['image'].dtype, sample['label'])

    plt.imshow(val_dataset[1]['image'])

    # x, y = load_data(input_file, '16x16')

    # Lets save all images from this file
    # Each image will be 3600x3600 pixels (10 000) images

    blank_image = None
    curr_index = 0
    image_index = 0

    # print('First image in dataset:')
    # print(x[curr_index].shape)
