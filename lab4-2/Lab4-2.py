#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import h5py
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import torch

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(10)


# In[25]:


# Calculate the value of mean and varience of this dataset


# In[8]:


import cv2
 
img_h, img_w = 512, 512   
means, stdevs = [], []
img_list = []
 
imgs_path = r'./test/'
imgs_path_list = os.listdir(imgs_path)
 
len_ = len(imgs_path_list)
i = 0

for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i,'/',len_)    

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.
 
for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))


# In[2]:


# Dataloader


# In[2]:


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        
        #self.file = h5py.File('data.h5', 'r')
        #self.preloadimg = self.file['train_img'][[i for i in range(35126)],:,:]
        
        self.transformations = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),
                                                 #transforms.RandomResizedCrop((512, 512)),
                                                 transforms.RandomRotation(degrees=30),
                                                 transforms.ToTensor(), 
                                                 transforms.Normalize((0.3775, 0.2618, 0.1873),(0.2918, 0.2088, 0.1684))
                                               ])     
        
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)
        
    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        single_img_name = os.path.join(self.root, self.img_name[index]+'.jpeg')
        single_img = Image.open(single_img_name)  # read an PIL image
        #single_img = Image.fromarray(self.preloadimg[index])
        img = self.transformations(single_img)
        label = self.label[index]

        return img, label
    


# In[3]:


# ResNet18


# In[3]:


class ResNet18(nn.Module):
    def __init__(self, num_class, pretrained=False):
        """
        Args:
            num_class: #target class
            pretrained: 
                True: the model will have pretrained weights, and only the last layer's 'requires_grad' is True(trainable)
                False: random initialize weights, and all layer's 'require_grad' is True
        """
        #super(ResNet18, self).__init__()
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)   
        
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False   # freezing the part of the model as no changes happen to its parameters.
                                            
        num_ftrs = self.model.fc.in_features   
        self.model.fc = nn.Linear(num_ftrs, num_class)   # Replace the last fully-connected layer
        
    def forward(self,X):
        out = self.model(X)
        return out
    


# In[6]:


resnet18 = models.resnet18(pretrained=False)
print(resnet18)


# In[5]:


# ResNet50


# In[4]:


class ResNet50(nn.Module):
    def __init__(self, num_class, pretrained=False):
        """
        Args:
            num_class: #target class
            pretrained: 
                True: the model will have pretrained weights, and only the last layer's 'requires_grad' is True(trainable)
                False: random initialize weights, and all layer's 'require_grad' is True
        """
        #super(ResNet50,self).__init__()
        super().__init__()
        self.model = models.resnet50(pretrained=pretrained)
        
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False 
                
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_class)
        
    def forward(self,X):
        out = self.model(X)
        return out
    


# In[7]:


resnet50 = models.resnet50(pretrained=False)
print(resnet50)


# In[5]:


def train(model, loader_train, loader_test, Loss,optimizer, epochs, device, num_class,name):
    """
    Args:
        model: resnet model
        loader_train: training dataloader
        loader_test: testing dataloader
        Loss: loss function
        optimizer: optimizer
        epochs: number of training epoch
        device: gpu/cpu
        num_class: #target class
        name: model name when saving model
    Returns:
        dataframe: with column 'epoch','acc_train','acc_test'
    """
    
    model.to(device)
    df = pd.DataFrame()
    df['epoch'] = range(1, epochs+1)
    best_model_wts = None
    best_evaluated_acc = 0
    acc_train = list()
    acc_test = list()
    
    for epoch in range(1, epochs+1):
        """train"""
        with torch.set_grad_enabled(True):
            model.train()
            total_loss = 0
            correct = 0
            for images, targets in loader_train:
                images, targets = images.to(device), targets.to(device,dtype=torch.long)
                predict = model(images)
                loss = Loss(predict, targets)
                total_loss += loss.item()
                correct += predict.max(dim=1)[1].eq(targets).sum().item()              
                """update"""
                optimizer.zero_grad() #重設參數梯度(gradient)
                loss.backward() 
                optimizer.step() #更新參數
                
            total_loss /= len(loader_train.dataset)
            acc = 100. * correct / len(loader_train.dataset)
            acc_train.append(acc)
            print(f'epoch{epoch:>2d} loss:{total_loss:.4f} acc:{acc:.2f}%')
            
        """evaluate"""
        _, acc = evaluate(model, loader_test, device, num_class)
        acc_test.append(acc)
        # update best_model_wts
        if acc > best_evaluated_acc:
            best_evaluated_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    df['acc_train'] = acc_train
    df['acc_test'] = acc_test
    
    # save model
    torch.save(best_model_wts, os.path.join('models', name+'.pt'))
    model.load_state_dict(best_model_wts)
    
    return df


# In[6]:


def evaluate(model, loader_test, device, num_class):
    """
    Args:
        model: resnet model
        loader_test: testing dataloader
        device: gpu/cpu
        num_class: #target class
    Returns:
        confusion_matrix: (num_class,num_class) ndarray
        acc: accuracy rate
    """
    
    confusion_matrix = np.zeros((num_class, num_class))
    
    with torch.set_grad_enabled(False):
        model.eval()
        correct = 0
        for images, targets in loader_test:  
            images, targets = images.to(device), targets.to(device,dtype=torch.long)
            predict = model(images)
            predict_class = predict.max(dim=1)[1]
            correct += predict_class.eq(targets).sum().item()
            
            for i in range(len(targets)):
                confusion_matrix[int(targets[i])][int(predict_class[i])] += 1
                
        acc = 100. * correct / len(loader_test.dataset)
        
    # normalize confusion_matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1).reshape(num_class, 1)
    
    return confusion_matrix, acc


# In[7]:


def plot(dataframe1, dataframe2, title):
    """
    Arguments:
        dataframe1: dataframe with 'epoch','acc_train','acc_test' columns of without pretrained weights model 
        dataframe2: dataframe with 'epoch','acc_train','acc_test' columns of with pretrained weights model 
        title: figure's title
    Returns:
        figure: an figure
    """
    
    fig = plt.figure(figsize=(10,6))
    
    for name in dataframe1.columns[1:]:
        plt.plot(range(1,1+len(dataframe1)), name,data=dataframe1, label=name[4:]+'(no pretraining)')
        
    for name in dataframe2.columns[1:]:
        plt.plot(range(1,1+len(dataframe2)), name,data=dataframe2, label=name[4:]+'(with pretraining)')
        
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.title(title)
    plt.legend()
    
    return fig

def plot_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.xaxis.set_label_position('top')
    
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(i, j, '{:.2f}'.format(confusion_matrix[j, i]), va='center', ha='center')
            
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    
    return fig


# In[8]:


# ResNet18


# In[13]:


batch_size = 64   
lr = 1e-3 
momentum = 0.9   
weight_decay = 5e-4   
Loss = nn.CrossEntropyLoss()
epochs = 20   # 10(resnet18), 5(resnet50)
epochs_feature_extraction = 5
epochs_fine_tuning = 15
num_class = 5


# In[14]:


dataset_train = RetinopathyLoader(root='data', mode='train')
loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)

dataset_test = RetinopathyLoader(root='data', mode='test')
loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True)


# In[15]:


"""no pretrained weights"""
model_no_pretrained = ResNet18(num_class=num_class, pretrained=False)
optimizer = optim.SGD(model_no_pretrained.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
#optimizer = optim.RMSprop(model_no_pretrained.parameters(), lr=lr, weight_decay=weight_decay)
df_no_pretrained = train(model_no_pretrained, loader_train, loader_test, Loss, optimizer, epochs, device,
                         num_class, 'resnet18_no_pretraining')

# test and save confusion matrix figure
confusion_matrix, _ = evaluate(model_no_pretrained, loader_test, device, num_class)
figure = plot_confusion_matrix(confusion_matrix)
figure.savefig('ResNet18(no pretrained weights).png')


"""with pretrained weights (feature extraction for few epochs, then finefuning for some epochs)"""
model_with_pretrained = ResNet18(num_class=num_class, pretrained=True)

# feature extraction
params_to_update=[]
for name, param in model_with_pretrained.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
df_firststep = train(model_with_pretrained, loader_train, loader_test, Loss, optimizer, epochs_feature_extraction,
                     device, num_class, 'resnet18_with_pretraining')
# finetuning
for param in model_with_pretrained.parameters():
    param.requires_grad = True
optimizer = optim.SGD(model_with_pretrained.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
df_secondstep = train(model_with_pretrained, loader_train, loader_test, Loss, optimizer, epochs_fine_tuning,
                      device, num_class, 'resnet18_with_pretraining')
df_with_pretrained = pd.concat([df_firststep, df_secondstep], axis=0, ignore_index=True)

# test and save confusion matrix figure
confusion_matrix, _ = evaluate(model_with_pretrained, loader_test, device, num_class)
figure = plot_confusion_matrix(confusion_matrix)
figure.savefig('ResNet18(with pretrained weights).png')

# plot accuracy figure
figure = plot(df_no_pretrained, df_with_pretrained, 'Result Comparison(ResNet18)')
figure.savefig('Result Comparison(ResNet18).png')


# In[17]:


print('resnet18_no_pretraining')
print(df_no_pretrained)

print('resnet18_with_pretraining')
print(df_with_pretrained)


# In[ ]:


# ResNet50


# In[18]:


batch_size=16


# In[19]:


dataset_train = RetinopathyLoader(root='data',mode='train')
loader_train = DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True)

dataset_test = RetinopathyLoader(root='data',mode='test')
loader_test = DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=False)


# In[20]:


"""no pretrained weights"""
model_no_pretrained = ResNet50(num_class=num_class, pretrained=False)
optimizer = optim.SGD(model_no_pretrained.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
df_no_pretrained = train(model_no_pretrained, loader_train, loader_test, Loss, optimizer, epochs, device,
                         num_class, 'resnet50_no_pretraining')

# test and save confusion matrix figure
confusion_matrix, _ = evaluate(model_no_pretrained, loader_test, device, num_class)
figure = plot_confusion_matrix(confusion_matrix)
figure.savefig('ResNet50(no pretrained weights).png')


"""with pretrained weights (feature extraction for few epochs, then finefuning for some epochs)"""
model_with_pretrained = ResNet50(num_class=num_class, pretrained=True)

# feature extraction
params_to_update=[]
for name, param in model_with_pretrained.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
df_firststep = train(model_with_pretrained, loader_train, loader_test, Loss, optimizer, epochs_feature_extraction,
                     device, num_class, 'resnet50_with_pretraining')
# finetuning
for param in model_with_pretrained.parameters():
    param.requires_grad = True
optimizer = optim.SGD(model_with_pretrained.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
df_secondstep = train(model_with_pretrained, loader_train, loader_test, Loss, optimizer, epochs_fine_tuning,
                      device, num_class, 'resnet50_with_pretraining')
df_with_pretrained = pd.concat([df_firststep, df_secondstep], axis=0, ignore_index=True)

# test and save confusion matrix figure
confusion_matrix, _ = evaluate(model_with_pretrained, loader_test, device, num_class)
figure = plot_confusion_matrix(confusion_matrix)
figure.savefig('ResNet50(with pretrained weights).png')

# plot accuracy figure
figure = plot(df_no_pretrained, df_with_pretrained, 'Result Comparison(ResNet50)')
figure.savefig('Result Comparison(ResNet50).png')


# In[21]:


print('resnet50_no_pretraining')
print(df_no_pretrained)

print('resnet50_with_pretraining')
print(df_with_pretrained)


# In[ ]:




