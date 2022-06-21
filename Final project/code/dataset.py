import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import librosa
import librosa.display
import matplotlib.pyplot as plt



default_transform = transforms.Compose([transforms.ToTensor()])

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
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                for d3 in os.listdir('%s/%s/%s' % (self.data_dir, d1, d2)):
                    self.dirs.append('%s/%s/%s/%s' % (self.data_dir, d1, d2, d3))

        # print(self.dirs)

        self.transformations = transform
        # self.fig = plt.figure()


    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        # print(self.dirs[index])
        y , sr = librosa.load(self.dirs[index], sr=16000)
        M1 = librosa.feature.melspectrogram(y=y, n_fft=512, n_mels=64, sr=sr)
        M_db1 = librosa.power_to_db(M1, ref=np.max)
        # print(M_db1.shape)

        # Show image
        fig = plt.figure()
        plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
        plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
        plt.axis('off')
        librosa.display.specshow(M_db1, x_axis='time', y_axis='mel')

        # Get RGB image array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        w, h = img.shape[0], img.shape[1]
        img = Image.fromarray(img, mode='RGB')
        # img = img.resize((int(h/2), int(w/2)), Image.ANTIALIAS)
        img = img.resize((64, 64), Image.ANTIALIAS)
        img = self.transformations(img)
        plt.cla()
        plt.close("all")
        # plt.clf()

        # M_db1_tensor = self.transformations(M_db1)

        # return M_db1_tensor
        return img

