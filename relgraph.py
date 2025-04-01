import os
import pickle

import igraph
import numpy as np
from tqdm import tqdm
import sys
import utils
from grapher import Grapher, IndexGraph
import argparse
import random
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from relgraph_util import create_relation_graph, generate_relation_graph,get_relation_triplets, create_static_relation_graph

parser = argparse.ArgumentParser(description='Relation Graph')
parser.add_argument('--dataset', type=str, default='icews0515',
                    help='name of the dataset')
parser.add_argument('--output_dir', type=str, default=None,
                    help='directory to store output')
parser.add_argument('--data_dir', type=str, default='data/',
                    help='directory to load data')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--train_history_len', type=int, default=10,
                    help='length of historical graphs')
parser.add_argument('--training_mode', type=str, default='train',
                    help='whether train, valid or test')

args = parser.parse_args()
print(args)
random.seed(args.seed)
dataset = args.dataset
data_dir = "data/" + dataset + "/"
if dataset in ['WIKI','YAGO']:
    data = IndexGraph(data_dir)
else:
    data = Grapher(data_dir)

if args.training_mode == 'train':
    train_data = data.train_idx
elif args.training_mode == 'valid':
    train_data = data.valid_idx
else:
    train_data = data.test_idx
if args.data_dir is None:
    data_dir = "data/"
else:
    data_dir = args.data_dir
if args.output_dir is None:
    output_dir = f'data/graph_data/{args.dataset}/'
else:
    output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_list = data.all_idx
data_list = utils.split_by_time_with_t(data_list, np.unique(data_list[:, 3]))
train_list = utils.split_by_time_with_t(train_data, np.unique(train_data[:, 3]))
time_list = np.unique(train_data[:,3])
num_rel = len(data.inv_relation_id)
A =0
def softmax(x):

    max = np.max(x, axis=1)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

def l2_normalization(matrix,axis):
    row_norms = np.linalg.norm(matrix, axis=axis)
    row_norms +=0.01
    if axis==0:
        return matrix / row_norms
    return matrix / row_norms[:, np.newaxis]


B = create_relation_graph(data_list[0:args.train_history_len], len(data.entity2id), len(data.relation2id_old))
# B = l2_normalization(B.todense(), 0)
A += l2_normalization(B.todense(), 1)

for up_idx in tqdm(time_list[args.train_history_len:]):
    idx = up_idx - args.train_history_len
    history_data = data_list[idx:up_idx]

    B = generate_relation_graph(history_data,len(data.entity2id),len(data.relation2id_old))
    # B = l2_normalization(B.todense(),0)
    A += l2_normalization(B.todense(),1)

A = l2_normalization(A,1)

with open(output_dir+'relation_graph.pickle','wb') as f:
    pickle.dump(A,f)