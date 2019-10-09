from model import TCN

import torch
from torch import nn
# import torch.utils.tensorboard as tensorboard

from TCN.tcn import TemporalConvNet

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset, TensorDataset

import time
import argparse
import torch.optim as optim
import torch.nn.functional as F

from  torch.distributions.categorical import Categorical

# matplotlib.use('Agg')

from torch.utils.tensorboard import SummaryWriter
import gym
import gym_Vibration

from model import TCN
from TCN.tcn import TemporalConvNet


torch.cuda.current_device()
torch.cuda._initialized = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

parser = argparse.ArgumentParser(description='Sequence Modeling - forecasting')
parser.add_argument("--env_name", default="VibrationEnv-v0")  # OpenAI gym environment name  VibrationEnv  Pendulum
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--seq_len', type=int, default=1,
                    help='sequence length (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=30,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()


class NormalizedActions(gym.ActionWrapper):
    def action(self, a):
        l = self.action_space.low
        h = self.action_space.high

        a = l + (a + 1.0) * 0.5 * (h - l)
        a = np.clip(a, l, h)

        return a

    def reverse_action(self, a):
        l = self.action_space.low
        h = self.action_space.high

        a = 2 * (a -l)/(h - l) -1 
        a = np.clip(a, l, h)

        return a

env = NormalizedActions(gym.make(args.env_name))


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

input_channels = state_dim
output_channels = action_dim

num_channels = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
seq_len = args.seq_len

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, seq_len):
        super(Actor, self).__init__()
        # self.tcn = TCN(state_dim, state_dim, num_channels, kernel_size=kernel_size, dropout=dropout).to(device)  # 预测所有状态
        self.bn1 = nn.BatchNorm1d(state_dim)
        self.tcn = TemporalConvNet(input_channels, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(num_channels[-1], state_dim) 
        self.fc2 = nn.Linear(num_channels[-1], action_dim)

    def forward(self, x):

        x = x.reshape(-1, state_dim, seq_len)
        x = self.bn1(x)
        x = self.tcn(x)
        # print(x.size())
        next_state_pred = F.relu(self.fc1(x[:, :, -1]))
        action = F.relu(self.fc2(x[:, :, -1]))

        return next_state_pred, action

    # def prediction(self, x):
    #     x = x.reshpe(-1, state_dim, seq_len)
    #     x = self.tcn(x)
    #     return x

    # def select_action(self, x):
    #     # x = torch.FloatTensor(x).to(device)
    #     return 100*x  #100 中间质量块的质量




if __name__ == "__main__":
 
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    input_channels = 6
    output_size = 1
    batch_size = args.batch_size
    seq_length = args.seq_len
    epochs = args.epochs


    print(args)
    print("Producing data...")

    # Note: We use a very simple setting here (assuming all levels have the same # of channels.
    num_channels = [args.nhid]*args.levels
    kernel_size = args.ksize
    dropout = args.dropout
    # model = TCN(input_channels, output_size, num_channels, kernel_size=kernel_size, dropout=dropout)
    model = Actor(state_dim, action_dim, seq_len).to(device)
    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    # hh = torch.rand(1,6,1)
    # s, a = model(hh)
    # print('s: {}, a: {}'.format(s.size(), a.size()))
    # print('s: {}, a: {}'.format(s, a))

    next_state_real = env.reset()
    next_state_real = torch.from_numpy(next_state_real).float().to(device)
    print('00000000',next_state_real.size())

    '''
    def train(epoch):
        global lr 
        global next_state_real
        batch_idx = 1
        total_loss = 0
        model.train()
        for i in range(10):
            optimizer.zero_grad()
            next_state_pred, probs = model(next_state_real)
            # print(' i= {}, probs = {}, size = {}'.format(i, probs, probs.size()))         

            m = Categorical(probs + 1e-10)
            # print(m)
            # print(' i= {}, m = {}'.format(i, m))  
            action = m.sample()
            next_state_real, reward, done, info  = env.step(action.detach().numpy())
            
            loss = -m.log_prob(action) * torch.from_numpy(reward).float().to(device)
   
            next_state_real = torch.from_numpy(next_state_real).float().to(device)
            # print('$$$$$$',next_state_real[-1])
            # loss = F.mse_loss(0e1 * torch.Tensor([1]), next_state_real[-1])
            # print('######', loss)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            # batch_idx += 1
            total_loss += loss.item()
            print('i: {}, loss: {}'.format(i, loss.item()))

            # if batch_idx % args.log_interval == 0:
            #     cur_loss = total_loss / args.log_interval
            #     processed = min(i+batch_size, train_x.size(0))
            #     print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
            #         epoch, processed, train_x.size(0), 100.*processed/train_x.size(0), lr, cur_loss))
            #     total_loss = 0
        '''

    def train(epoch):
        global lr 
        global next_state_real
        batch_idx = 1
        total_loss = 0
        # model.train()
        model.eval()
        for _ in range(1000):
            optimizer.zero_grad()
            next_state_pred, probs = model(next_state_real)
            # print(' i= {}, probs = {}, size = {}'.format(i, probs, probs.size()))         

            action = probs
            next_state_real, reward, done, info  = env.step(action.detach().cpu().numpy())
            next_state_real = torch.from_numpy(next_state_real).float().to(device)
            a = 0 * torch.tensor([1.], requires_grad=True).to(device)
            b = next_state_real[-1] * torch.tensor([1.], requires_grad=True).to(device)
            loss = F.mse_loss(a, b)
            # loss = torch.rand([1,2])
            # print('i: {}, loss: {}, {}, {}'.format(i, loss, loss.size(), type(loss)) )
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            batch_idx += 1
            total_loss += loss.item()
            # print('epoch: {}, loss: {}'.format(epoch, loss.item()))

            if batch_idx % args.log_interval == 0:
                cur_loss = total_loss / args.log_interval
                
                print('Train Epoch: {:2d} \tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                    epoch, lr, cur_loss))
                # processed = min(i+batch_size, train_x.size(0))
                # print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                #     epoch, processed, train_x.size(0), 100.*processed/train_x.size(0), lr, cur_loss))
                total_loss = 0

    for ep in range(1, epochs+1):
        train(ep)