import torch
import numpy as np
import pandas as pd
import argparse
import os
from utils.io import load_yaml
import yaml
from toymodels.TWD_Model import TWD
from toymodels.TWD_TO_Model import TWD_TO
from toymodels.OWD_NI_Model import OWD_NI
from toymodels.OWD_NT_Model import OWD_NT
from toymodels.TWD_MLP_Model import TWD_MLP

parser = argparse.ArgumentParser(description='toy case eval')
parser.add_argument('--d_seed', type=int, default=11, metavar='N',
                    help='dataset seed')
parser.add_argument('--d_number', type=int, default=250, metavar='N',
                    help='dataset trajectories')
parser.add_argument('--d_t', type=int, default=50, metavar='N',
                    help='dataset T')
parser.add_argument('--e_degree', type=float, default=1.0, metavar='N',
                    help='environment degree')
parser.add_argument('--c_degree', type=float, default=1.0, metavar='N',
                    help='confounding degree')
parser.add_argument('--GPU', type=str, default='cuda:0')
parser.add_argument('--cuda', default=False, action='store_true')

parser.add_argument('--h_dim', type=int, default=32, metavar='N',
                    help='h_dim')
parser.add_argument('--z_dim', type=float, default=32, metavar='N',
                    help='z_dim')
parser.add_argument('--alpha', type=float, default=1, metavar='N',
                    help='alpha')
parser.add_argument('--embedding_size', type=float, default=2, metavar='N',
                    help='embedding_size')
parser.add_argument('--batchsize', type=int, default=64, metavar='N',
                    help='batchsize')
parser.add_argument('--l_rate', type=float, default=0.001, metavar='N',
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='N',
                    help='weight_decay')
parser.add_argument('--random', default=True, action='store_true')
parser.add_argument('--size', type=float, default=0.3, metavar='N',
                    help='random policy')
parser.add_argument('--method', type=str, default='TWD', metavar='N',
                    help='different methods')
parser.add_argument('--MCs', type=int, default=10, metavar='N',
                    help='the number of MC')

args = parser.parse_args()
method = args.method
np.random.seed(args.d_seed)
torch.manual_seed(args.d_seed)
torch.cuda.manual_seed(args.d_seed)
torch.set_num_threads(1)
path = "results/" + method + '/'
if not os.path.exists(path):
    os.makedirs(path)
    print(path + ' 创建成功')

if args.random:
    save_dir = "results/" + method + '/OPE' + str(args.size) + method + str(args.d_seed) + str(args.d_number) + str(
        args.d_t) \
               + str(args.e_degree) + str(args.c_degree) + str(args.batchsize) + str(args.l_rate) + '.csv'
else:
    save_dir = "results/" + method + '/OPES3' + method + str(args.d_seed) + str(args.d_number) + str(args.d_t) \
               + str(args.e_degree) + str(args.c_degree) + str(args.batchsize) + str(args.l_rate) + '.csv'

data_savedir = 'dataseed/simdata' + str(args.d_seed) + str(args.d_number) + str(args.d_t) + str(
    args.e_degree) + str(args.c_degree) + '.csv'
data = pd.read_csv(data_savedir)

u, t = args.d_number, args.d_t
h_dim = load_yaml('tables/toy/op_hyper_params.yml', key='toy')[
    str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(args.e_degree) + '_' + str(
        args.c_degree) + '_tune'+method]['h_dim']

z_dim = load_yaml('tables/toy/op_hyper_params.yml', key='toy')[
    str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(args.e_degree) + '_' + str(
        args.c_degree) + '_tune'+method]['z_dim']

alpha = load_yaml('tables/toy/op_hyper_params.yml', key='toy')[
    str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(args.e_degree) + '_' + str(
        args.c_degree) + '_tune'+method]['alpha']

embedding_size = load_yaml('tables/toy/op_hyper_params.yml', key='toy')[
    str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(args.e_degree) + '_' + str(
        args.c_degree) + '_tune'+method]['embedding_size']

batch_size = load_yaml('tables/toy/op_hyper_params.yml', key='toy')[
    str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(args.e_degree) + '_' + str(
        args.c_degree) + '_tune'+method]['batch_size']

learning_rate = load_yaml('tables/toy/op_hyper_params.yml', key='toy')[
    str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(args.e_degree) + '_' + str(
        args.c_degree) + '_tune'+method]['learning_rate']

w_decay = load_yaml('tables/toy/op_hyper_params.yml', key='toy')[
    str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(args.e_degree) + '_' + str(
        args.c_degree) + '_tune'+method]['w_decay']

lamda = load_yaml('tables/toy/op_hyper_params.yml', key='toy')[
    str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(args.e_degree) + '_' + str(
        args.c_degree) + '_tune'+method]['lamda']

if method == 'TWD':
    model = TWD(input_dim=5, output_dim=10, n_u=u, n_t=t, h_dim=h_dim, z_dim=z_dim, alpha=alpha,
                embedding_size=embedding_size, act_type='elu')
elif method == 'OWD_NI':
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

model.eval()
path = 'modelresults/' + method + '/' + method + str(args.d_seed) + str(args.d_number) + str(args.d_t) \
       + str(args.e_degree) + str(args.c_degree) + str(h_dim) + str(z_dim) + str(alpha) + str(embedding_size) + str(
    batch_size) + str(learning_rate) + str(w_decay) + str(lamda) + 'net_params' + '.pth'
saved_state_dict = torch.load(path)  # 加载模型参数
model.load_state_dict(saved_state_dict)  # 应用到网络中

#MC
data1 = data.loc[(data.ct == 0), :]
x = data1.iloc[:, 3:7]
s0 = np.array(x)
s0 = torch.tensor(s0, dtype=torch.float32)
s0 = s0.view(-1, 4)
reward = 0
MCs = args.MCs
for mc_seed in range(MCs):
    np.random.seed(mc_seed)
    torch.manual_seed(mc_seed)
    torch.cuda.manual_seed(mc_seed)
    #s0
    s = torch.tensor(s0, dtype=torch.float32)
    s = s.view(-1, 4)
    cu = torch.tensor(np.arange(u), dtype=torch.int64)
    cu = cu.view(-1, 1)
    for j in range(t):
        ct = torch.tensor(np.full(u, j), dtype=torch.int64)
        ct = ct.view(-1, 1)
        if args.random:
            ap = torch.tensor(np.full(u, args.size), dtype=torch.float32)
        else:
            pass
        a = torch.distributions.bernoulli.Bernoulli(ap)
        a = a.sample()
        output1, output2m, output2v = model(cu, ct, s, a)
        cov = torch.diag_embed(torch.exp(output2v))
        mean = output2m
        output = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
        output = output.sample()
        s = output[:, 1:5].view(-1, 4)
        s = s.clamp(-1e3, 1e3)
        reward = reward + np.array((output[:, 0].clamp(-1e3, 1e3)).detach().numpy())

print(np.mean(reward, axis=0) / MCs)
dataframe = pd.DataFrame({'OPEval': [round(np.mean(reward, axis=0) / MCs, 4)]})
dataframe.to_csv(save_dir, index=True, sep=',')
table_path = load_yaml('config/global.yml', key='path')['tables']
hyper_params_dict = yaml.safe_load(open(table_path + 'toy/' + 'op_hyper_params.yml', 'r'))
if args.random:
    hyper_params_dict['toy'][
        str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(args.e_degree) + '_' + str(
            args.c_degree) + '_tune' + method]['result' + str(args.size)] = round(float(np.mean(reward, axis=0) / MCs),
                                                                                  4)
    yaml.dump(hyper_params_dict, open(table_path + 'toy/' + 'op_hyper_params.yml', 'w'), default_flow_style=False)
else:
    hyper_params_dict['toy'][
        str(args.d_seed) + '_' + str(args.d_number) + "_" + str(args.d_t) + "_" + str(args.e_degree) + '_' + str(
            args.c_degree) + '_tune' + method]['result'] = round(float(np.mean(reward, axis=0) / MCs), 4)
    yaml.dump(hyper_params_dict, open(table_path + 'toy/' + 'op_hyper_params.yml', 'w'), default_flow_style=False)
