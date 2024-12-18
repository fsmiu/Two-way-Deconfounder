import numpy as np
import pandas as pd
import argparse
import os

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sim(s0,d_number,d_t,e_degree,c_degree):
    ls = list()
    for i in range(d_number):
        x = s0[i, :]
        for j in range(d_t):
            z1 = cu[i] * ct[j]
            z2 = cu[i] - ct[j]
            z3 = cu[i] + ct[j]
            z4 = -cu[i] - ct[j]
            z5 = -cu[i]+ct[j]
            ap = sigmoid(1 * (0.25 * (x[0] + x[1] + x[2] + x[3])-4 + 3 * c_degree * (z3+z1)))
            print(ap)
            a = np.random.binomial(1, ap, 1)[0]
            next_x0 = np.random.normal((0.8) * x[0] + 0.1*c_degree*z2+2*a-0.5, e_degree)
            next_x1 = np.random.normal((0.8) * x[1] + 0.1*c_degree*z3+2*a-0.5, e_degree)
            next_x2 = np.random.normal((0.8) * x[2] + 0.1*c_degree*z4+2*a-0.5, e_degree)
            next_x3 = np.random.normal((0.8) * x[3] + 0.1*c_degree*z5+2*a-0.5, e_degree)
            r = np.random.normal((0.25 * (x[0] + x[1] + x[2] + x[3]) + 3 * c_degree*(z1+z3) + 2.5 * a), e_degree)
            ls.append(list([i, j, x[0], x[1], x[2], x[3], a, cu[i], ct[j], r, next_x0, next_x1, next_x2, next_x3]))
            x[0] = next_x0
            x[1] = next_x1
            x[2] = next_x2
            x[3] = next_x3

    return ls

parser = argparse.ArgumentParser(description='toy case simulation')
parser.add_argument('--d_seed', type=int, default=11, metavar='N',
                    help='dataset_seed')
parser.add_argument('--d_t_seed', type=int, default=11, metavar='N',
                    help='dataset_t_seed')
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
args = parser.parse_args()

#path
path_cu = 'cuseed/'
path_ct = 'ctseed/'
if not os.path.exists(path_cu):
    os.makedirs(path_cu)
    print(path_cu + 'success')
if not os.path.exists(path_ct):
    os.makedirs(path_ct)
    print(path_ct + 'success')
cu_savedir = 'cuseed/' + 'cu' + str(args.d_seed) + '.csv'
ct_savedir = 'ctseed/' + 'ct' + str(args.d_seed) + '.csv'
#seed
np.random.seed(args.d_t_seed)
ct = np.random.normal(0, 1, size=args.d_t)
np.random.seed(args.d_seed)
cu = np.random.normal(0, 1, size=args.d_number)
np.savetxt(cu_savedir, cu, delimiter=",")
np.savetxt(ct_savedir, ct, delimiter=",")
##s0
mean = (0, 0, 0, 0)
alpha = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
s0 = np.random.multivariate_normal(mean, alpha, size=args.d_number)
#dataset
ls=sim(s0,args.d_number,args.d_t,args.e_degree,args.c_degree)
name = ['cu', 'ct', 'x0', 'x1', 'x2', 'x3', 'a', 'confounder_u', 'confounder_t', 'r', 'next_x0', 'next_x1', 'next_x2',
        'next_x3']
simdata = pd.DataFrame(columns=name, data=ls)
ls=np.array(ls)

path_dataseed = 'dataseed/'
if not os.path.exists(path_dataseed):
    os.makedirs(path_dataseed)
    print(path_dataseed + 'success')
data_savedir = 'dataseed/simdata'+str(args.d_seed)+str(args.d_number)+str(args.d_t)+str(args.e_degree) + str(args.c_degree) + '.csv'
simdata.to_csv(data_savedir, encoding='utf-8')