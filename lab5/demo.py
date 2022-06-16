import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, plot_rec, finn_eval_seq, pred, plot_psnr, plot_kl, mse_metric, plot_result, gif_generate
from PIL import Image

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='./model', help='base directory to save logs')
    # parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')

    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    # parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    # parser.add_argument('--epoch_size', type=int, default=100, help='epoch size')

    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=120, help='The epoch that teacher forcing ratio become decreasing')
    # parser.add_argument('--tfr_start_decay_epoch', type=int, default=70, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0.5, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=True, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=2, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=5, help='The number of cycle for kl annealing (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=12, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')

    parser.add_argument('--cond_dim', type=int, default=7, help='dimensionality of condition one-hot vector')

    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  


    args = parser.parse_args()
    return args

def train(x, cond, modules, optimizer, kl_anneal, args, epoch):

    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0
    use_teacher_forcing = True if random.random() < args.tfr else False
    # print(args.tfr)

    x = x.float()

    h_seq = [modules['encoder'](x[:,i]) for i in range(args.n_past+args.n_future)]

    train_result = []
    origin_result = []

    for i in range(1, args.n_past + args.n_future):
        # raise NotImplementedError

        h_target = h_seq[i][0]

        if args.last_frame_skip or i < args.n_past:	
            h, skip = h_seq[i-1]
        else:
            h = h_seq[i-1][0]
         
        if i > 1:
            previous_img = x_pred
            pr_h = modules['encoder'](previous_img)
            h_no_tfr = pr_h[0]
        else:
            h_no_tfr = h

        z_t, mu, logvar = modules['posterior'](h_target)

        c = cond[:, i, :].float()

        # h_pred = modules['frame_predictor'](torch.cat([h, z_t, c], 1))

        if use_teacher_forcing:
            h_pred = modules['frame_predictor'](torch.cat([h, z_t, c], 1))
        else:
            # print("without teacher")
            h_pred = modules['frame_predictor'](torch.cat([h_no_tfr, z_t, c], 1))

        x_pred = modules['decoder']([h_pred, skip])
        mse += nn.MSELoss()(x_pred, x[:,i])
        # mse += mse_metric(x_pred.detach().cpu().numpy(), x[:,i].detach().cpu().numpy())
        kld += kl_criterion(mu, logvar, args)

        # train_result.append(x_pred.data.cpu().numpy())
        # origin_result.append(x[:,i].data.cpu().numpy())

    beta = kl_anneal.get_beta(epoch)
    # print('B: ', beta)
    loss = mse + kld * beta
    # loss = mse + kld * args.beta
    loss.backward()

    # train_result = np.array(train_result)
    # tt = (np.transpose(train_result[1,1,:,:], (1, 2, 0)) + 1) * 255.0
    # data = Image.fromarray(np.uint8(tt))
    # data.save('./psnr_gen/psnr_'+str(epoch)+'_gt.png')

    # origin_result = np.array(origin_result)
    # tt = (np.transpose(origin_result[1,1,:,:], (1, 2, 0)) + 1) * 255.0
    # data = Image.fromarray(np.uint8(tt))
    # data.save('./psnr_gen/psnr_'+str(epoch)+'.png')

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past), beta
    

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        # raise NotImplementedError

        self.time = args.niter / args.kl_anneal_cycle
        # print(args.epoch_size, args.kl_anneal_cycle)
        # print(self.time)

        if not args.kl_anneal_cyclical:
            self.mode = 'monotonic'
            print('monotonic')
        else:
            self.mode = 'cyclical'
    
    # def update(self):
    #     raise NotImplementedError

    def get_beta(self, epoch):
        # raise NotImplementedError      

        # print(epoch, self.time)
        if self.mode == 'monotonic':
            return (1./(self.time))*(epoch) if epoch<self.time else 1.
        else:
            # period = args.epoch_size//self.time
            epoch %= self.time
            # KL_weight = sigmoid((epoch - period // 2) / (period // 10)) / 2
            KL_weight = (1./(self.time/2))*(epoch) if epoch<self.time/2 else 1.
            return KL_weight
        


def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        print("go!!!")
        saved_model = torch.load('%s/model_lp.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+args.cond_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    print('device:', device)
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    test_data = bair_robot_pushing_dataset(args, 'test')

    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)

    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    test_iterator = iter(test_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)
    # kl_anneal = 0

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------

    psnr_list = []
    PSNR = []
    seq, cond = None, None

    for _ in range(len(test_data) // args.batch_size):
        try:
            test_seq, test_cond = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_loader)
            test_seq, test_cond = next(test_iterator)
        
        test_seq, test_cond = test_seq.type(torch.FloatTensor),test_cond.type(torch.FloatTensor)
        test_seq, test_cond = test_seq.to(device),test_cond.to(device)

        pred_seq = pred(test_seq, test_cond, modules, args, device)
        
        _, _, psnr = finn_eval_seq(test_seq[:,args.n_past:], pred_seq[:])
        psnr_list.append(psnr)

        ave_psnr = np.mean(np.concatenate(psnr))
        PSNR.append(ave_psnr)

        with open('./result_seq/test_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('test psnr = {:.5f} \n'.format(ave_psnr)))


    print('Best PSNR: ', max(PSNR))
    print('Average PSNR: ', sum(PSNR) / len(PSNR))

    ori_list = np.array(test_seq[0].cpu())
    pre_list = np.array(pred_seq[0])
    result_pre = Image.new('RGB', (64*12, 64))

    for i in range(12):  
        result = Image.new('RGB', (128, 64))       
        ori = (np.transpose(ori_list[i,:,:,:], (1, 2, 0)) + 1) *255
        data1 = Image.fromarray(np.uint8(ori))

        if i < 2:
            pre = (np.transpose(ori_list[i,:,:,:], (1, 2, 0)) + 1) *255
        else:
            pre = (np.transpose(pre_list[i-2,:,:,:], (1, 2, 0)) + 1) *255
             
        data2 = Image.fromarray(np.uint8(pre))
        result.paste(data1, box=(0, 0))
        result.paste(data2, box=(64, 0))
        result_pre.paste(data2, box=(0+64*i, 0))
        result.save('./result_seq/compare_'+str(i)+'.png')

    result_pre.save('./result_seq/'+'prediction.png')
    fp_in = "./result_seq/compare_*.png"
    fp_out = "./result_seq/result_compare.gif"
    gif_generate(fp_in, fp_out)




if __name__ == '__main__':
    main()
        
