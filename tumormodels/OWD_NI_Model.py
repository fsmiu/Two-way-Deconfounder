import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

def init_weights(m):
    ''' 初始化模型权重 '''
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = (t < mean - 2 * std) | (t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(
                cond,
                torch.nn.init.normal_(torch.ones(t.shape),
                                      mean=mean,
                                      std=std), t)
        return t

    if type(m) == nn.Linear:
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(m._input_dim)))
        m.bias.data.fill_(0.0)


class OWD_NI(nn.Module):
    def __init__(self, input_dim, output_dim, n_u, n_t, h_dim, z_dim, alpha, embedding_size=4, act_type='elu'):
        super(OWD_NI, self).__init__()
        #dim
        self.s = 4
        self.r = 1
        self.sa = 5
        self.a = 2
        # embedding
        self.T = nn.Embedding(n_t, embedding_size)
        # activation function
        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        else:
            raise RuntimeError('unknown act_type {0}'.format(act_type))
        '''z net'''
        self.z_predictor = nn.Sequential()
        self.z_predictor.add_module('z_fc1', nn.Linear(input_dim - self.a + embedding_size, h_dim))
        self.z_predictor.add_module('z_act1', self.act)
        self.z_predictor.add_module('z_fc2', nn.Linear(h_dim, h_dim))
        self.z_predictor.add_module('z_act2', self.act)

        '''a net'''
        self.a_predictor = nn.Sequential()
        self.a_predictor.add_module('a_fc1', nn.Linear(h_dim, h_dim // 2))
        self.a_predictor.add_module('a_act1', self.act)
        self.a_predictor.add_module('a_fc2', nn.Linear(h_dim // 2, self.a))
        self.a_predictor.add_module('a_sig2', nn.Sigmoid())
        '''y net'''
        self.y_predictor = nn.Sequential()
        self.y_predictor.add_module('y_fc1', nn.Linear(h_dim + self.a, h_dim // 2))
        self.y_predictor.add_module('y_act1', self.act)
        self.y_predictor.add_module('y_fc2', nn.Linear(h_dim // 2, output_dim))

        self._max_logvar = nn.Parameter((torch.ones(
            (1, int(output_dim / 2))).float() * 5),
                                        requires_grad=False)
        self._min_logvar = nn.Parameter((-torch.ones(
            (1, int(output_dim / 2))).float() * 10),
                                        requires_grad=False)
        self.alpha = alpha
        self.output_dim = output_dim

    def z_net(self, nto):
        z = self.z_predictor(nto)
        return z
    def a_net(self, z):
        a_prob = self.a_predictor(z)
        return a_prob
    def y_net(self, z, a):
        x = torch.cat((z, a), 1)
        y = self.y_predictor(x)
        return y

    def forward(self, n_id, t_id, x_o, x_a):
        # n_e = self.N(torch.tensor(n_id, dtype=torch.int64, requires_grad=False)).squeeze()
        t_e = self.T(torch.tensor(t_id, dtype=torch.int64, requires_grad=False)).squeeze()
        NTO = torch.cat((t_e, x_o), 1)
        z = self.z_net(NTO)
        y1 = self.a_net(z)
        y2 = self.y_net(z, x_a.reshape(len(x_a), -1))
        y2m = y2[:, :int(self.output_dim / 2)]
        y2v = self._max_logvar - F.softplus(self._max_logvar - y2[:, int(self.output_dim / 2):])
        y2v = self._min_logvar + F.softplus(y2v - self._min_logvar)

        return y1, y2m, y2v
