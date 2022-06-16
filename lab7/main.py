import os
import torch
from torch.utils.data import DataLoader

from datahelper import CLEVRDataset
# from model_ori import Generator, Discriminator
from model import Generator, Discriminator, weights_init
# from model3 import Generator, Discriminator, weights_init

import torch.nn as nn
import numpy as np
import copy

from util import get_test_conditions,save_image
from evaluator import evaluation_model

manualSeed = 1233
#manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
torch.manual_seed(manualSeed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 100
c_dim = 200
image_shape = (64, 64, 3)
epochs = 200
lr = 0.0001
batch_size = 64


def train(dataloader, g_model, d_model, z_dim, epochs, lr):
    """
    :param z_dim: 100
    """

    Criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(g_model.parameters(), lr, betas=(0.5,0.99))
    optimizer_d = torch.optim.Adam(d_model.parameters(), lr, betas=(0.5,0.99))
    eval_model = evaluation_model()

    test_conditions = get_test_conditions(os.path.join('dataset','test.json')).to(device)
    fixed_z = torch.randn(len(test_conditions), z_dim).to(device)
    best_score = 0

    for epoch in range(1, 1+epochs):
        total_loss_g = 0
        total_loss_d = 0

        for i, (images,conditions) in enumerate(dataloader):
            g_model.train()
            d_model.train()

            batch_size = len(images)
            images = images.to(device)
            conditions = conditions.to(device)

            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)
            """
            train discriminator
            """
            optimizer_d.zero_grad()

            # for real images
            predicts = d_model(images, conditions)
            loss_real = Criterion(predicts, real)
            # for fake images
            z = torch.randn(batch_size, z_dim).to(device)
            gen_imgs = g_model(z, conditions)
            predicts = d_model(gen_imgs.detach(), conditions)
            loss_fake = Criterion(predicts, fake)
            # bp
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            """
            train generator
            """
            for _ in range(4):
                optimizer_g.zero_grad()

                z = torch.randn(batch_size, z_dim).to(device)
                gen_imgs = g_model(z, conditions)
                predicts = d_model(gen_imgs, conditions)
                loss_g = Criterion(predicts, real)
                # bp
                loss_g.backward()
                optimizer_g.step()

            print(f'epoch{epoch} {i}/{len(dataloader)}  loss_g: {loss_g.item():.3f}  loss_d: {loss_d.item():.3f}')
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()

        # evaluate
        g_model.eval()
        d_model.eval()
        with torch.no_grad():
            gen_imgs = g_model(fixed_z, test_conditions)
        score = eval_model.eval(gen_imgs, test_conditions)
        if score>best_score:
            best_score = score
            best_model_wts = copy.deepcopy(g_model.state_dict())
            torch.save(best_model_wts, os.path.join('models', f'epoch{epoch}_score{score:.2f}.pt'))
        print(f'avg loss_g: {total_loss_g/len(dataloader):.3f}  avg_loss_d: {total_loss_d/len(dataloader):.3f}')
        print(f'testing score: {score:.2f}')
        print('---------------------------------------------')
        # savefig
        save_image(gen_imgs, os.path.join('results', f'epoch{epoch}.png'), nrow=8, normalize=True)


if __name__=='__main__':

    # load training data
    dataset_train = CLEVRDataset(img_path='dataset/iclevr', json_path=os.path.join('dataset','train.json'))
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    # create generate & discriminator
    generator = Generator(z_dim, c_dim).to(device)
    discrimiator = Discriminator(image_shape, c_dim).to(device)
    # generator.weight_init(mean=0, std=0.02)
    # discrimiator.weight_init(mean=0, std=0.02)

    generator.apply(weights_init)
    discrimiator.apply(weights_init)

    # train
    train(loader_train, generator, discrimiator, z_dim, epochs, lr)