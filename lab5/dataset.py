import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from imageio import *



default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        #raise NotImplementedError

        self.seq_len = 12
        self.image_size = 64
        self.seed_is_set = False # multi threaded loading
        self.d = 0
        self.qq = 0

        self.root_dir = args.data_root

        if mode=='train':
            self.data_dir = '%s/train' % self.root_dir
            self.ordered = False
        elif mode=='validate':
            self.data_dir = '%s/validate' % self.root_dir
            self.ordered = True
        else:
            self.data_dir = '%s/test' % self.root_dir
            self.ordered = True

        self.dirs = []

        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))

        self.transformations = transform


    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)


    def __len__(self):
        # raise NotImplementedError
        #return self.seq_len
        return len(self.dirs)

        
    def get_seq(self):
        #raise NotImplementedError

        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d += 1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]

        image_seq = []

        for i in range(self.seq_len):
            fname = '%s/%d.png' % (d, i)
            im = Image.open(fname)  # read an PIL image
            img = np.array(im).reshape(1, 64, 64, 3)/255.
            image_seq.append(img)

        image_seq = np.concatenate(image_seq, axis=0)
        #image_seq = self.transformations(image_seq)
        image_seq = torch.from_numpy(image_seq)
        image_seq = image_seq.permute(0,3,1,2)
        #print(image_seq.shape)
        self.qq = d

        return image_seq
        
    
    def get_csv(self):
        # raise NotImplementedError

        d = self.qq
        path_a = '%s/actions.csv' %d
        path_p = '%s/endeffector_positions.csv' %d

        with open(path_a, newline='') as csvfile:
            rows = csv.reader(csvfile)
            action = np.asarray(list(rows))
            #print(action.shape)

        with open(path_p, newline='') as csvfile:
            rows = csv.reader(csvfile)
            position = np.asarray(list(rows))
            #print(position.shape)

        c = np.concatenate((action,position),axis=1)
        c = c.astype(np.float)
        #print(c.shape)

        return c

    
    def __getitem__(self, index):
        self.set_seed(index)
        # seq = self.transformations(self.get_seq())
        seq  = self.get_seq()
        # cond = self.transformations(self.get_csv())
        cond = self.get_csv()
        return seq, cond




