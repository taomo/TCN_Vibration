'''
time: 2019.10.5 night  

'''

from model import TCN

import torch
from torch import nn
# import torch.utils.tensorboard as tensorboard


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
# matplotlib.use('Agg')

cuda = torch.cuda.is_available()

usecols=[0, 1, 2, 3, 4, 5]
k = 1 # 向前的时刻
n = len(usecols) # 多少个变量


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def ts_dataframe_to_supervised(df, target, n_in=1, n_out=1, dropT=True):
    """
    Transform a time series dataframe into a supervised learning dataset.
    Arguments:
        df: a dataframe.
        target: this is the target variable you intend to use in supervised learning
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropT: Boolean - whether or not to drop columns at time "t".
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    namevars = df.columns.tolist()
    # input sequence (t-n, ... t-1)
    drops = []
    for i in range(n_in, -1, -1):
        if i == 0:
            for var in namevars:
                addname = var+'(t)'
                df.rename(columns={var:addname},inplace=True)
                drops.append(addname)
        else:
            for var in namevars:
                addname = var+'(t-'+str(i)+')'
                df[addname] = df[var].shift(i)
    # forecast sequence (t, t+1, ... t+n)
    if n_out == 0:
        n_out = False
    for i in range(1, n_out):
        for var in namevars:
            addname = var+'(t+'+str(i)+')'
            df[addname] = df[var].shift(-i)
    # drop rows with NaN values
    df.dropna(inplace=True,axis=0)
    # put it all together
    target = target+'(t)'
    if dropT:
        drops.remove(target)
        df.drop(drops, axis=1, inplace=True)
    preds = [x for x in list(df) if x not in [target]] 
    return df, target, preds

def load_data(file_name, sequence_length=10, split=0.8):
       
    # load dataset
    # df = pd.read_csv(file_name, sep=',', usecols=[1])
    # names =  [r'$x1(t)$', r'$x2(t)$', r'$x1c(t)$', r'$x2c(t)$', r'$x1cc(t)$', r'$x2cc(t)$']
    # usecols=[0, 1, 2, 3, 4, 5]
    # usecols=[5]
    df = pd.read_csv(file_name,header=None,sep=',',  skiprows=1, usecols=usecols)
  
    print(df)

   # load dataset
    values = df.values
    # print('values',values)
    # ensure all data is float
    values = values.astype('float')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # print('scaled',scaled)
    # frame as supervised learning
    # k = 1 # 向前的时刻
    # n = len(usecols) # 多少个变量
    # target = 'var6'
    # df, target,preds = ts_dataframe_to_supervised(df, target, 1, 0, False)

    reframed = series_to_supervised(scaled, k, 1)

    # print(type(reframed))
    # print(reframed.size)

    split_boundary = int(reframed.shape[0] * split)

    #fram
    train_x = reframed.ix[: split_boundary, :k*n]
    # train_x = train_x.values.reshape(6,-1)
    # print('train_x',train_x)
    test_x = reframed.ix[split_boundary:, :k*n]
    # print(test_x)
    train_y = reframed.ix[: split_boundary, k*n:]
    # print('train_y',train_y)
    test_y = reframed.ix[split_boundary:, k*n:]

    train = reframed.ix[: split_boundary, :]
    test = reframed.ix[split_boundary:, :]    

    print('reframed',reframed.head())
    # print('train_x',train_x.head().T)
    # print('train_y',train_y.head().T)
    # print('train_x',train_x.head())
    

    return train, test, train_x, train_y, test_x, test_y, scaler    











# 创建TCN网络



if __name__ == "__main__":
    
    # 导入数据

    # datapath = "D:\\AI\GitHub\\tcn-timeseries-pytorch\\taomo.csv"
    # datapath = "D:\\AI\GitHub\\tcn-timeseries-pytorch\\taomo10_1k"
    datapath = "D:\\AI\GitHub\\tcn-timeseries-pytorch\\taomo10k.csv"


    train_data, test_data, train_x, train_y, test_x, test_y, scaler  = load_data(datapath)
    
    train_x = torch.from_numpy(train_x.values).float()
    train_x = train_x.reshape(-1,n,k)  # 10 # 向前的时刻  6个变量

    # A =  np.arange(60)
    # A = A.reshape(-1,10)
    # print(A)
    # A = A.reshape(-1,5)
    # print(A)

    train_y = torch.from_numpy(train_y.values).float()
    train_y = train_y[:, -1]

    test_x = torch.from_numpy(test_x.values).float()
    test_x = test_x.reshape(-1,n,k)
    print(test_x.size())


    test_y = torch.from_numpy(test_y.values).float()
    test_y = test_y[:, -1]
    # 数据变形 准备
    # train_x = train_x.unsqueeze(1)
    train_y = train_y.unsqueeze(1) 

    # test_x = test_x.unsqueeze(1)
    test_y = test_y.unsqueeze(1) 

    # train = torch.from_numpy(train.values).float()
    # train = train.unsqueeze(1)    
    # test = torch.from_numpy(test.values).float()

    print(train_x.size())
    print(train_y.size())
    



    parser = argparse.ArgumentParser(description='Sequence Modeling - forecasting')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (default: 0.0)')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: -1)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit (default: 10)')
    parser.add_argument('--ksize', type=int, default=7,
                        help='kernel size (default: 7)')
    parser.add_argument('--levels', type=int, default=8,
                        help='# of levels (default: 8)')
    parser.add_argument('--seq_len', type=int, default=400,
                        help='sequence length (default: 400)')
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
    model = TCN(input_channels, output_size, num_channels, kernel_size=kernel_size, dropout=dropout)

    # a = torch.rand(3,6,10)
    # aa = model(train_x)
    # print(aa)

    if args.cuda:
        model.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()

    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)




    def train(epoch):
        global lr
        model.train()
        batch_idx = 1
        total_loss = 0
        for i in range(0, train_x.size(0), batch_size):
            if i + batch_size > train_x.size(0):
                x, y = train_x[i:], train_y[i:]
            else:
                x, y = train_x[i:(i+batch_size)], train_y[i:(i+batch_size)]
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            batch_idx += 1
            total_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                cur_loss = total_loss / args.log_interval
                processed = min(i+batch_size, train_x.size(0))
                print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                    epoch, processed, train_x.size(0), 100.*processed/train_x.size(0), lr, cur_loss))
                total_loss = 0



    def draw(yi, color):
        plt.plot(np.arange(test_x.size(0)), yi[:test_x.size(0)], color, linewidth = 2.0)


    def evaluate(ep):
        model.eval()
        with torch.no_grad():
            output = model(test_x)
            test_loss = F.mse_loss(output, test_y)
            print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))

        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Redlines are predicted values, epoch:{},loss:{})'.format(ep,test_loss.item()), fontsize=30)
        plt.xlabel('t', fontsize=20)
        plt.ylabel('Vibration', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)     
        draw(output.data.cpu().numpy(), 'r')
        # plt.legend("test loss:%d"%loss.item(),loc='lower left')
        # plt.legend("test loss:",loc='lower left')
        real_test_y = test_y.data.cpu().numpy()
        draw(real_test_y, 'b')
        plt.grid()
        plt.savefig('./output/predict{}.pdf'.format(ep))
        plt.close()
        # plt.show()

        return test_loss.item()


    for ep in range(1, epochs+1):
        train(ep)
        tloss = evaluate(ep)