import numpy as np
import pandas as pd
import argparse
import os

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

parser = argparse.ArgumentParser(description='toy case true')

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
parser.add_argument('--random', default=True, action='store_true')
parser.add_argument('--size', type=float, default=0.3, metavar='N',
                    help='random degree')
parser.add_argument('--MC', type=int, default=10000, metavar='N',
                    help='MC number')

args = parser.parse_args()
np.random.seed(args.d_seed)
path = "results/" + 'MCTure/'
if not os.path.exists(path):
    os.makedirs(path)
    print(path + ' success')
if args.random:
    save_dir = "results/" + 'MCTure' + '/OPE' + str(args.size) + 'MCture' + str(args.d_seed) + \
               str(args.e_degree) + str(args.c_degree) + '.csv'
else:
    save_dir = "results/" + 'MCTure' + '/OPES3' + 'MCture' + str(args.d_seed) + \
               str(args.e_degree) + str(args.c_degree) + '.csv'

mean = (0, 0, 0, 0)
alpha = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
s0 = np.random.multivariate_normal(mean, alpha, size=args.MC)
#cu_savedir = 'cuseed/' + 'cu' + str(args.d_seed) + '.csv'
ct_savedir = 'ctseed/' + 'ct' + str(args.d_seed) + '.csv'
#cu = np.loadtxt(cu_savedir, delimiter=',')
ct = np.loadtxt(ct_savedir, delimiter=',')
cu = np.random.normal(0, 1, size=args.MC)
ls = list()
for i in range(len(s0)):
    x = s0[i, :]
    for j in range(len(ct)):
        z1 = cu[i] * ct[j]
        z2 = cu[i] - ct[j]
        z3 = cu[i] + ct[j]
        z4 = -cu[i] - ct[j]
        z5 = -cu[i] + ct[j]
        if args.random:
            ap = args.size
        else:
            ap = sigmoid(0.3 * (0.25 * (x[0] + x[1] + x[2] + x[3]) + 0.2))
        a = np.random.binomial(1, ap, 1)[0]
        next_x0 = np.random.normal((0.8) * x[0] + 0.1 * args.c_degree * z2 + 2 * a - 0.5, args.e_degree)
        next_x1 = np.random.normal((0.8) * x[1] + 0.1 * args.c_degree * z3 + 2 * a - 0.5, args.e_degree)
        next_x2 = np.random.normal((0.8) * x[2] + 0.1 * args.c_degree * z4 + 2 * a - 0.5, args.e_degree)
        next_x3 = np.random.normal((0.8) * x[3] + 0.1 * args.c_degree * z5 + 2 * a - 0.5, args.e_degree)

        r = np.random.normal(1 * (0.25 * (x[0] + x[1] + x[2] + x[3]) + 3 * args.c_degree * (z1 + z3) + 2.5 * a),
                             args.e_degree)
        ls.append(list([i, j, r]))
        x[0] = next_x0
        x[1] = next_x1
        x[2] = next_x2
        x[3] = next_x3

ls = np.array(ls)
MCtrue=ls.sum(axis=0) / args.MC
print('OPEval',MCtrue[2])
dataframe = pd.DataFrame({'OPEval': [round((ls.sum(axis=0)[-1] / args.MC), 4)]})
dataframe.to_csv(save_dir, index=True, sep=',')