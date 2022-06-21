import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from PIL import Image


root_dir = './data/train/fan'
img_dir = './data_image/train/fan_test'

def scale_minmax(X, min=0.0, max=1.0):
    """
    Minmax scaler for a numpy array
    
    PARAMS
    ======
        X (numpy array) - array to scale
        min (float) - minimum value of the scaling range (default: 0.0)
        max (float) - maximum value of the scaling range (default: 1.0)
    """
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


for d1 in os.listdir(root_dir):
    p1 = img_dir + '/%s' % (d1)
    # print('p1', p1)

    for d2 in os.listdir('%s/%s' % (root_dir, d1)):
        p2 = p1 + '/%s' % (d2)
        # print('p2', p2)
        if not os.path.exists(p2):
            os.makedirs(p2)

        for d3 in os.listdir('%s/%s/%s' % (root_dir, d1, d2)):
            data = '%s/%s/%s/%s' % (root_dir, d1, d2, d3)
            name = d3.split('.')
            # print(name)

            y, sr = librosa.load(data, sr=16000)
            # # M1 = librosa.feature.melspectrogram(y=y, n_fft=512, n_mels=64, sr=sr)
            # # M_db1 = librosa.power_to_db(M1, ref=np.max)
            # M_db1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64)

            # fig = plt.figure()
            # plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
            # plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
            # plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
            # plt.axis('off')
            # librosa.display.specshow(M_db1, x_axis='time', y_axis='mel')

            # fig.savefig(p2 + '/%s' % (name[0]) + '.png')
            # plt.cla()
            # plt.close("all")

            mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=512)
            mels = librosa.power_to_db(mels, ref=np.max)

            # Preprocess the image: min-max, putting 
            # low frequency at bottom and inverting to 
            # match higher energy with black pixels:
            img = scale_minmax(mels, 0, 255).astype(np.uint8)
            img = np.flip(img, axis=0)
            img = 255 - img
            img = Image.fromarray(img)

            # Saving the picture generated to disk:
            img.save(p2 + '/%s' % (name[0]) + '.png')

print('Done!')


### Test ###
# y , sr = librosa.load('./data/train/valve/id_06/normal/00000120.wav', sr=16000)
# M1 = librosa.feature.melspectrogram(y=y, n_fft=512, n_mels=64, sr=sr)
# M_db1 = librosa.power_to_db(M1, ref=np.max)

# fig = plt.figure()
# plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
# plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
# plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
# plt.axis('off')
# librosa.display.specshow(M_db1, x_axis='time', y_axis='mel')

# fig.savefig('./test.png')