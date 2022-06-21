import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

import librosa
import librosa.display
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt



default_transform = transforms.Compose([
    # transforms.Resize([64, 64]),
    transforms.ToTensor()])

class mimii_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'

        self.root_dir = args.data_root  #'./data'
        self.data_type = args.data_type  #'valve'

        if mode=='train':
            self.data_dir = '%s/train/%s' % (self.root_dir, self.data_type)
            self.ordered = False
        elif mode=='validate':
            self.data_dir = '%s/val/%s' % (self.root_dir, self.data_type)
            self.ordered = True
        else:
            self.data_dir = '%s/test/%s' % (self.root_dir, self.data_type)
            self.ordered = True

        self.dirs = []

        for d1 in os.listdir(self.data_dir):
            if not d1 == 'id_06':
                continue
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                for d3 in os.listdir('%s/%s/%s' % (self.data_dir, d1, d2)):
                    self.dirs.append('%s/%s/%s/%s' % (self.data_dir, d1, d2, d3))

        # print(self.dirs)

        self.transformations = transform


    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        # print(self.dirs[index])

        # Get RGB image array
        # fig.canvas.draw()
        # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # w, h = img.shape[0], img.shape[1]
        # img = Image.fromarray(img, mode='RGB')
        # # img = img.resize((int(h/2), int(w/2)), Image.ANTIALIAS)
        # img = img.resize((64, 64), Image.ANTIALIAS)
        # img = self.transformations(img)
        # plt.cla()
        # plt.close("all")
        # plt.clf()

        fname = self.dirs[index]
        # img = Image.open(fname)  # read an PIL image
        # print(img.size)
        img = cv2.imread(fname)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        img = Image.fromarray(img, mode='RGB')
        img = img.resize((64, 64), Image.ANTIALIAS)
        img = self.transformations(img)
        # print(img.shape)

        return img

