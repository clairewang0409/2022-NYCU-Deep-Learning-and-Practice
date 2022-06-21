import os
import torch
import argparse
import datetime
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from IPython.core.display import Image, display
from random import randint
import time
from tqdm import tqdm

import utils
# from dataset import mimii_dataset
from dataset_image import mimii_dataset
from model_vae_32 import VAE



def main(args):

    with open('./{}/train_record.txt'.format(args.result_path), 'a') as train_record:
        train_record.write('args: {}\n\n'.format(args))

    train_dataset = mimii_dataset(args, 'train')
    # val_dataset = mimii_dataset(args, 'validate')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    ### generate the model
    if not args.load_model == '':
        print("Load pretrained model!")
        model = torch.load(args.load_model + '/model_' + args.data_type + '.pt').to(device)
    else:
        model = VAE().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr) 

    ### Fixed input for debugging
    # fixed_x = next(iter(train_loader)).to(device)
    # save_image(fixed_x, 'real_image.png')
    # Image('real_image.png')

    print('Start training...')

    progress = tqdm(total=args.epochs)

    best_loss = 0.0

    for epoch in range(args.epochs):

        model.train()
        epoch_loss = 0.0

        start = time.time()

        for i, images in enumerate(train_loader):
            # try:
            #     x = next(iter(train_loader)).to(device)
            # except StopIteration:
            #     train_iterator = iter(train_loader)
            #     x = next(train_iterator).to(device)

            # print('images: ', images.shape)
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss, bce, kld = utils.loss_fn(recon_images, images, mu, logvar, args.beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()/args.batch_size

            # if i % 20 == 0:
            #     # print('x: ', x.shape)
            #     recon_x, _, _ = model(x)
            #     compare_x = torch.cat([x, recon_x])
            #     save_image(compare_x.data.cpu(), args.result_path + f'/recon_image_{epoch}_{i}.png')

        # valiadate(model, val_loader)

        recon_x, _, _ = model(images)
        compare_x = torch.cat([images, recon_x])
        save_image(compare_x.data.cpu(), args.result_path + f'/recon_image_{epoch}.png')

        end = time.time()
        t = end - start

        progress.update(1)

        avg_epoch_loss = epoch_loss/len(train_loader)
        # print("Epoch [{}/{}]  Loss: {:.3f}  Time: {:.3f} s".format(epoch+1, args.epochs, epoch_loss/len(train_loader), t))

        if epoch == 0:
            best_loss = avg_epoch_loss

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model, args.result_path + '/model_' + args.data_type + '.pt')

        print("\nLoss: {:.3f} | Best_loss: {:.3f}".format(avg_epoch_loss, best_loss))

        with open('./{}/train_record.txt'.format(args.result_path), 'a') as train_record:
                train_record.write(('epoch: %03d | loss: %.3f (best loss: %.3f) | time: %.3f s\n'  % (epoch+1, best_loss, avg_epoch_loss, t)))


    # recon_x, _, _ = model(fixed_x)
    # compare_x = torch.cat([fixed_x, recon_x])
    # save_image(compare_x.data.cpu(), 'sample_image.png')
    # display(Image('sample_image.png', width=700, unconfined=True))

    # torch.save(model, args.result_path + '/model.pt')




if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', default='./data_image', help='root directory for data')
    parser.add_argument('--data_type', default='slider_test')

    parser.add_argument("--epochs", type=int, default=500, help="Epochs of half lr")
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--beta', type=int, default=0.001)

    parser.add_argument('--load_model', type=str, default='',help="Path of hdf5 file")
    # parser.add_argument('--load_model', type=str, default='./results/2022-06-16-02-08_slider_test',help="Path of hdf5 file")

    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    result_dir = './results/{}'.format(nowTime +'_'+ parser.parse_args().data_type)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    parser.add_argument('--result_path', type=str, default='{}'.format(result_dir), help='Path to save model')

    args = parser.parse_args()
    # print(args)

    main(args)