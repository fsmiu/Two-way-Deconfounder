import argparse
import time
import yaml
import random
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
from utils.io import load_yaml
from utils.MyDataSet import sim_DataSet
from toymodels.TWD_Model import TWD
from toymodels.TWD_TO_Model import TWD_TO
from toymodels.OWD_NI_Model import OWD_NI
from toymodels.OWD_NT_Model import OWD_NT
from toymodels.TWD_MLP_Model import TWD_MLP

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, save_resultpath, patience=10, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.save_resultpath = save_resultpath
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, train_vali):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, train_vali)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, train_vali)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, train_vali):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        path = os.path.join(self.save_path)
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        resultpath = os.path.join(self.save_resultpath)
        dataframe = pd.DataFrame(train_vali)
        dataframe.to_csv(resultpath, index=True, sep=',')

        self.val_loss_min = val_loss

    def best_vali(self):
        return -self.best_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def objective(single_trial):
    trial = single_trial
    # sample a set of hyperparameters.
    h_dim = trial.suggest_categorical('h_dim', [64])
    z_dim = trial.suggest_categorical('z_dim', [32])
    alpha = trial.suggest_categorical('alpha', [1])
    embedding_size = trial.suggest_categorical('embedding_size', [2, 4, 8])
    batch_size = trial.suggest_categorical('batch_size', [256,512, 1024, 2048, 4096])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001,0.005])
    w_decay = trial.suggest_categorical('w_decay', [1e-4, 1e-2])
    lamda = trial.suggest_categorical('lamda', [0.0, 0.3, 0.5, 0.7])

    #path
    net_save_dir = 'modelresults/' + method+'/' + method + str(args.d_seed) + str(args.d_number) + str(args.d_t) \
                   + str(args.e_degree) + str(args.c_degree) + str(h_dim) + str(z_dim) + str(alpha) + str(
        embedding_size) + str(batch_size) + str(learning_rate) + str(w_decay) + str(lamda) + 'net_params' + '.pth'
    train_save_dir = 'trainresults/' + method+'/' + method + str(args.d_seed) + str(args.d_number) + str(args.d_t) \
                     + str(args.e_degree) + str(args.c_degree) + str(h_dim) + str(z_dim) + str(alpha) + str(
        embedding_size) + str(batch_size) + str(learning_rate) + str(w_decay) + str(lamda) + 'trainresult' + '.csv'
    net_path = 'modelresults/' + method+'/'
    train_path = 'trainresults/' +method+'/'
    if not os.path.exists(net_path):
        os.makedirs(net_path)
        print(net_path + 'success')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        print(train_path + 'success')
    early_stopping = EarlyStopping(net_save_dir, train_save_dir)
    train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testDataset, batch_size=batch_size)
    # model
    if method=='TWD':
        model = TWD(input_dim=5, output_dim=10, n_u=u, n_t=t, h_dim=h_dim, z_dim=z_dim, alpha=alpha,
                       embedding_size=embedding_size, act_type='elu')
    elif method=='OWD_NI':
        model = OWD_NI(input_dim=5, output_dim=10, n_u=u, n_t=t, h_dim=h_dim, z_dim=z_dim, alpha=alpha,
                    embedding_size=embedding_size, act_type='elu')
    elif method == 'OWD_NT':
        model = OWD_NT(input_dim=5, output_dim=10, n_u=u, n_t=t, h_dim=h_dim, z_dim=z_dim, alpha=alpha,
                    embedding_size=embedding_size, act_type='elu')
    elif method == 'TWD_TO':
        model = TWD_TO(input_dim=5, output_dim=10, n_u=u, n_t=t, h_dim=h_dim, z_dim=z_dim, alpha=alpha,
                    embedding_size=embedding_size, act_type='elu')
    elif method == 'TWD_MLP':
        model = TWD_MLP(input_dim=5, output_dim=10, n_u=u, n_t=t, h_dim=h_dim, z_dim=z_dim, alpha=alpha,
                    embedding_size=embedding_size, act_type='elu')

    #train
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    train_loss, test_loss = [], []
    train_loss1, test_loss1 = [], []
    train_loss2, test_loss2 = [], []
    train_loss3, test_loss3 = [], []
    for epoch in range(num_epoch):
        train_loss_sum = 0.0
        train_loss1_sum = 0.0
        train_loss2_sum = 0.0
        train_loss3_sum = 0.0
        train_len = 0

        model.train()
        model.to(device)
        for i, data_ in enumerate(train_dataloader):
            x_u, x_m, x_s, x_a, ya, yrs = data_[0].to(device), data_[1].to(device), data_[2].to(device), data_[3].to(
                device), data_[4].to(device), data_[5].to(device)

            optimizer.zero_grad()
            output1, output2m, output2v = model(x_u, x_m, x_s, x_a)
            inverse_output2v = torch.exp(-output2v)
            mse_loss2 = torch.mean(torch.mean(torch.pow(output2m - yrs, 2) *
                                              inverse_output2v,
                                              dim=-1),
                                   dim=-1)
            var_loss2 = torch.mean(torch.mean(output2v, dim=-1), dim=-1)
            if args.method=='TWD_TO':
                loss1=mse_loss2
                loss2 = mse_loss2 + var_loss2
                loss =  loss2
            else:
                loss1 = criterion(output1, ya.reshape(-1, 1))
                loss2 = mse_loss2 + var_loss2
                loss = lamda * loss1 + (1 - lamda) * loss2

            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_loss1_sum += loss1.item()
            train_loss2_sum += mse_loss2.item()
            train_loss3_sum += var_loss2.item()
            train_len += 1
        print('epoch:', epoch, 'train_loss:', train_loss_sum / (train_len), "train_loss1:",
              train_loss1_sum / (train_len),
              "train_loss2:", train_loss2_sum / (train_len),
              "train_loss3:", train_loss3_sum / (train_len))
        train_loss.append(train_loss_sum / train_len)
        train_loss1.append(train_loss1_sum / train_len)
        train_loss2.append(train_loss2_sum / train_len)
        train_loss3.append(train_loss3_sum / train_len)

        # test
        model.eval()
        model.to(device)
        with torch.no_grad():
            test_loss_sum = 0.0
            test_loss1_sum = 0.0
            test_loss2_sum = 0.0
            test_loss3_sum = 0.0
            test_len = 0
            for i, data_ in enumerate(test_dataloader):
                x_u, x_m, x_s, x_a, ya, yrs = data_[0].to(device), data_[1].to(device), data_[2].to(device), data_[
                    3].to(device), data_[4].to(device), data_[5].to(device)
                output1, output2m, output2v = model(x_u, x_m, x_s, x_a)
                inverse_output2v = torch.exp(-output2v)
                mse_loss2 = torch.mean(torch.mean(torch.pow(output2m - yrs, 2) *
                                                  inverse_output2v,
                                                  dim=-1),
                                       dim=-1)
                var_loss2 = torch.mean(torch.mean(output2v, dim=-1), dim=-1)
                if args.method == 'TWD_TO':
                    loss1 = mse_loss2
                    loss2 = mse_loss2 + var_loss2
                    loss = loss2
                else:
                    loss1 = criterion(output1, ya.reshape(-1, 1))
                    loss2 = mse_loss2 + var_loss2
                    loss =  loss2
                test_loss_sum += loss.item()
                test_loss1_sum += loss1.item()
                test_loss2_sum += mse_loss2.item()
                test_loss3_sum += var_loss2.item()
                test_len += 1
            print('epoch:', epoch, 'test_loss:', test_loss_sum / (test_len), "test_loss1:", test_loss1_sum / (test_len),
                  "test_loss2:", test_loss2_sum / (test_len),
                  "test_loss3:", test_loss3_sum / (test_len))
            test_loss.append(test_loss_sum / test_len)
            test_loss1.append(test_loss1_sum / test_len)
            test_loss2.append(test_loss2_sum / test_len)
            test_loss3.append(test_loss3_sum / test_len)
            train_vali = {'epoch': [epoch],
                          "train_loss": [train_loss_sum / (train_len)],
                          "train_loss1": [train_loss1_sum / (train_len)],
                          "train_loss2": [train_loss2_sum / (train_len)],
                          "train_loss3": [train_loss3_sum / (train_len)],
                          "test_loss": [test_loss_sum / (test_len)],
                          "test_loss1": [test_loss1_sum / (test_len)],
                          "test_loss2": [test_loss2_sum / (test_len)],
                          "test_loss5": [test_loss3_sum / (test_len)]}
            early_stopping(test_loss_sum / test_len, model, train_vali)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
    best_valid_metric = early_stopping.best_vali()

    return best_valid_metric



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='toy train')
    parser.add_argument('--d_seed', type=int, default=11, metavar='N',
                        help='dataset seed')
    parser.add_argument('--d_number', type=int, default=1000, metavar='N',
                        help='dataset trajectories')
    parser.add_argument('--d_t', type=int, default=50, metavar='N',
                        help='dataset T')
    parser.add_argument('--e_degree', type=float, default=1.0, metavar='N',
                        help='environment degree')
    parser.add_argument('--c_degree', type=float, default=1.0, metavar='N',
                        help='confounding degree')
    parser.add_argument('--GPU', type=str, default='cuda:0')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--n_trials', type=int, default=300, metavar='N',
                        help='n_trials')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='train epochs')
    parser.add_argument('--method', type=str, default='TWD', metavar='N',
                        help='different methods')

    args = parser.parse_args()
    seed = args.d_seed
    n_trials = args.n_trials
    num_epoch = args.epochs
    method=args.method
    torch.set_num_threads(1)
    # device = torch.device(args.GPU if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device("cpu")
    # data
    setup_seed(seed)
    ##读取数据
    data_savedir = 'dataseed/simdata' + str(args.d_seed) + str(args.d_number) + str(args.d_t) + str(
        args.e_degree) + str(args.c_degree) + '.csv'
    data = pd.read_csv(data_savedir)
    x1, x2, y1, y2 = data.iloc[:, 1:3], data.iloc[:, 3:8], data.iloc[:, 7], data.iloc[:, 10:15]
    # 将DataFrame转为ndarray再转为Tensor
    x1 = torch.tensor(x1.values, dtype=torch.int64)
    x2 = torch.tensor(x2.values, dtype=torch.float32)
    x = torch.cat((x1, x2), 1)
    y1 = torch.tensor(y1.values, dtype=torch.float32)
    y2 = torch.tensor(y2.values, dtype=torch.float32)
    y1 = y1.view(-1, 1)
    y = torch.cat((y1, y2), 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2022)
    # users and length
    u, t = max(data.iloc[:, 1]) + 1, max(data.iloc[:, 2]) + 1
    print(u, t)
    # 构造数据集
    trainDataset = sim_DataSet(x_train[:, 0], x_train[:, 1], x_train[:, 2:6], x_train[:, 6], y_train[:, 0],
                                  y_train[:, 1:6])
    testDataset = sim_DataSet(x_test[:, 0], x_test[:, 1], x_test[:, 2:6], x_test[:, 6], y_test[:, 0], y_test[:, 1:6])

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    search_space = {'h_dim': [64], 'z_dim': [32], 'alpha': [1],
                    'embedding_size': [2, 4, 8],
                    'batch_size': [256,512, 1024, 2048, 4096],
                    'learning_rate': [0.001],
                    'w_decay': [1e-4, 1e-2],
                    'lamda': [0.0, 0.3, 0.5, 0.7]}
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe()
    best_params = study.best_params
    print(best_params)

    table_path = load_yaml('config/global.yml', key='path')['tables']
    if not os.path.exists(table_path + 'toy/'):
        os.makedirs(table_path + 'toy/')

    trials.to_csv(table_path + 'toy/' + str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(
        args.e_degree) + '_' + str(args.c_degree) + '_tune'+method+'.csv')

    if Path(table_path + 'toy/' + 'op_hyper_params.yml').exists():
        pass
    else:
        yaml.dump(dict(toy=dict()),
                  open(table_path + 'toy/' + 'op_hyper_params.yml', 'w'), default_flow_style=False)
    time.sleep(0.5)
    hyper_params_dict = yaml.safe_load(open(table_path + 'toy/' + 'op_hyper_params.yml', 'r'))
    hyper_params_dict['toy'][
        str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(args.e_degree) + '_' + str(
            args.c_degree) + '_tune'+method] = best_params
    yaml.dump(hyper_params_dict, open(table_path + 'toy/' + 'op_hyper_params.yml', 'w'), default_flow_style=False)
