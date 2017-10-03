from argparse import ArgumentParser
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_file', help="Input File with images")

    args = parser.parse_args()
    return args.in_file

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f)
    return dict

def load_data(input_file, input_size):
    d = unpickle(input_file)
    x = d['data']
    y = d['labels']

    x = np.dstack(x[:,:input_size], x[:, input_size:(2*input_size)],
        x[:, 2*input_size:])

    x = x.reshape((x.shape[0], 16, 16, 3))

    return x, y

if __name__ == '__main__':
    input_file, gen_images, hist_sorted  = parse_arguments()
    x, y = load_data(input_file, 512)

    # Lets save all images from this file
    # Each image will be 3600x3600 pixels (10 000) images

    blank_image = None
    curr_index = 0
    image_index = 0

    print('First image in dataset:')
    print(x[curr_index])
