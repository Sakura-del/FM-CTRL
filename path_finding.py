import pickle

import networkx as nx
from collections import defaultdict, Counter
import numpy as np
import random
from tqdm import tqdm
import sys
from colorama import Fore
import os
import argparse
import copy

import utils
from grapher import Grapher, IndexGraph

parser = argparse.ArgumentParser(description='Path Finding for Relation Prediction')
parser.add_argument('--dataset', type=str, default='YAGO',
                    help='name of the dataset')
parser.add_argument('--finding_mode', type=str, default='head',
                    help='whether head, relation or tail is fixed')
parser.add_argument('--training_mode', type=str, default='valid',
                    help='whether train, valid, test or interpret')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory to load data')
parser.add_argument('--output_dir', type=str, default=None,
                    help='directory to store output')
parser.add_argument('--train_dataset', type=str, default=None,
                    help='dataset to load train graph')
parser.add_argument('--sample_dataset', type=str, default=None,
                    help='dataset to load ranking files')
parser.add_argument('--npaths_ranking', type=int, default=3,
                    help='number of paths for each triplet')
parser.add_argument('--search_depth', type=int, default=4,
                    help='search depth')
parser.add_argument('--train_history_len', type=int, default=50,
                    help='length of historical graphs')
parser.add_argument('--neg_size', type=int, default=50,
                    help='number of negative samples')

args = parser.parse_args()
print(args)

G = nx.DiGraph()
train_triplets = []
ranking_triplets = []

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

relations = train_data[:, 1].tolist()
relation_count = Counter(list(range(len(data.relation2id))))
train_count = Counter(relations)
relation_count += train_count

relation_dict = data.id2relation
entity_dict = data.id2entity
entity_dict[-1] = ''
time_dict = data.id2ts

graph_file = "data/graph_data/"+dataset+"/relation_graph.pickle"
with open(graph_file,'rb') as f:
    graph_data = pickle.load(f)

if args.output_dir is None:
    if args.training_mode == 'interpret':
        output_dir = os.path.join('data/relation_prediction_path_data/', args.dataset,
                                  f"interpret/")
    else:
        output_dir = os.path.join('data/path_data/', args.dataset, f"ranking_{args.finding_mode}/")
else:
    output_dir = args.output_dir
if args.sample_dataset is None:
    sample_dataset = os.path.join("data/path_data/", args.dataset,
                                       f"ranking_{args.finding_mode}/sample_{args.training_mode}.pickle")
else:
    sample_dataset = args.sample_dataset

ranking_dataset = os.path.join("data/path_data/", args.dataset,
                              f"ranking_{args.finding_mode}/ranking_{args.training_mode}.pickle")

if args.dataset.split("-")[-1] == "inductive" and args.training_mode != 'train':
    graph = "inductive_graph.txt"
else:
    graph = "train_full.txt"

with open(sample_dataset, 'rb') as f:
    sample_quads = pickle.load(f)

def get_sentence(paths):
    text_paths = []
    for path in paths:
        relations = [relation_dict[j] for j in path[1::4]]
        times = [time_dict[i] for i in path[3::4]]
        entities = [entity_dict[j] for j in path[0::4]]

        p = [rv for r in zip(entities, relations, times) for rv in r]
        p.append(entity_dict[path[-2]])
        text_paths.append(p)
    return text_paths

def get_search_graph(neighbors, sub, ob, k_hop=2):
    try:
        sub_neighs = neighbors[sub][:, 2]
        ob_neighs = neighbors[ob][:, 2]
    except:
        return []
    if k_hop > 1:
        sub_neighs_2 = []
        ob_neighs_2 = []
        for s in sub_neighs:
            sub_neighs_2.append(neighbors[s][:, 2].tolist())
        for o in ob_neighs:
            ob_neighs_2.append(neighbors[o][:, 2].tolist())
        subgraph = set(sum(sub_neighs_2, [])) & set(sum(ob_neighs_2, []))
        subgraph.update(sub_neighs)
        subgraph.update(ob_neighs)
    else:
        subgraph = set(sub_neighs) & set(ob_neighs)
    if len(subgraph) == 0 and sub not in neighbors[ob][:, 2]:
        return {}
    if ob not in subgraph:
        subgraph.add(ob)

    return list(subgraph)

def generatePath(stack, edge):
    p = stack.copy()
    p.append(edge)
    p = p[1:]
    path = []
    for item in p[::-1]:
        path.append([item[2], data.inv_relation_id[item[1]], item[0], item[3]])
    path = np.hstack(path)
    return path

def edge_sample(neighbors,s,o,r):
    entities = neighbors.keys()
    if s in entities:
        s_edge = np.array(sorted(neighbors[s], key=lambda x: (-x[3],-graph_data.A[data.inv_relation_id[x[1]]][r])))[0]
        # s_edge = neighbors[s][np.random.choice(neighbors[s].shape[0],1)][0]
    else:
        s_edge = []
    if o in entities:
        o_edge = np.array(sorted(neighbors[o], key=lambda x: (-x[3],-graph_data.A[data.inv_relation_id[x[1]]][data.inv_relation_id[r]])))[0]
        # o_edge = neighbors[o][np.random.choice(neighbors[o].shape[0],1)][0]
        o_edge = o_edge[[2,1,0,3]]
        o_edge[1] = data.inv_relation_id[o_edge[1]]
    else:
        o_edge = []

    return np.hstack((s_edge,o_edge)).astype(int)

def findPathsHead(neighbors, input_quads, paths,ranking_quads, num_path,do_sample=False):
    for i in range(0,len(input_quads),args.neg_size):
        quads = input_quads[i:i+args.neg_size]
        search_paths = searchPaths(neighbors,quads[0],num_path,do_sample=do_sample)
        if len(search_paths) ==0:
            continue
        else:
            ranking_quads.extend(quads)
            paths.append(search_paths)
        for line in quads[1:]:
            search_paths = searchPaths(neighbors,line,num_path,do_sample=True)
            paths.append(search_paths)

def searchPaths(neighbors,line,num_path,do_sample=True,rel_graph=True):
    stack = []
    s, r, o, t = line
    visited = set([(line[2],line[3])])
    line = [s,data.inv_relation_id[r],o,t]
    stack.append(line)
    paths = []
    seen_path = defaultdict(list)
    subgraph = get_search_graph(neighbors, o, s,k_hop=1)
    if len(subgraph)==0:
        if do_sample:
            sample_edge = edge_sample(neighbors,s,o,r)
            if sample_edge.any():
                paths.append(sample_edge)
        return paths
    # seen_node=[]
    while (len(stack) > 0):
        _, rel, u, timestamp = stack[-1]
        # if u not in subgraph:
        if len(stack) >= args.search_depth:
            stack.pop()
            visited.remove((u,timestamp))
            continue
        g = 0
        if u not in seen_path.keys():
            seen_path[u] = neighbors[u][np.where(np.isin(neighbors[u][:,2],subgraph))]
            if rel_graph:
                seen_path[u] = np.array(sorted(seen_path[u], key=lambda x: (-x[3],-graph_data.A[x[1]][rel])))
            else:
                seen_path[u] = np.array(sorted(seen_path[u], key=lambda x: (-x[3],relation_count[x[1]])))
        if len(seen_path[u]) ==0:
            stack.pop()
            del seen_path[u]
            continue

        filterd_edges = seen_path[u][np.where(seen_path[u][:, 3] <= timestamp)]

        for edge in filterd_edges:
            if edge[2]==s:
                p = generatePath(stack, edge)
                paths.append(p)
                seen_path[u] = seen_path[u][~np.all(seen_path[u]==edge,axis=1)]
                filterd_edges = filterd_edges[~np.all(filterd_edges==edge,axis=1)]
            if len(paths) >= num_path:
                break
        if len(paths) >= num_path:
            break
        for edge in filterd_edges:
            try:
                if tuple(edge[2:]) not in visited and edge[2] not in seen_path.keys():
                    g = g + 1
                    stack.append(edge)
                    visited.add(tuple(edge[2:]))
                    seen_path[u] = seen_path[u][~np.all(seen_path[u]==edge,axis=1)]
                    break
            except:
                print(2)
        if g == 0:
            stack.pop()
            del seen_path[u]
    if len(paths)==0 and do_sample==True:
        sample_edge = edge_sample(neighbors,s,o,r)
        if sample_edge.any():
            paths.append(sample_edge)
    return paths

data_list = data.all_idx
data_list = np.vstack((data.train_idx,data.valid_idx,data.test_idx)) # only for acled_ind
data_list = utils.split_by_time_with_t(data_list,np.unique(data_list[:,3]))
input_list = utils.split_by_time_with_t(sample_quads, np.unique(sample_quads[:, 3]))
ranking_paths = []
ranking_quads = []
time_list = np.unique(train_data[:,3])
if args.training_mode=='train':
    train_idx = 1
else:
    train_idx = time_list[0]

for idx in tqdm(time_list):
    if idx < 1: continue
    if idx - args.train_history_len < 0:
        history_data = data_list[0:idx]
    else:
        history_data = data_list[idx - args.train_history_len:idx]
    history_data = np.vstack(history_data)
    neighbors = utils.store_history_graph(history_data)

    if args.finding_mode == 'head':
        findPathsHead(neighbors, input_list[idx-train_idx], ranking_paths,ranking_quads, args.npaths_ranking)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

with open(os.path.join(output_dir, f"paths_{args.training_mode}.pickle"), "wb") as f:
    pickle.dump(ranking_paths,f)

with open(ranking_dataset, 'wb') as f:
    pickle.dump(np.array(ranking_quads),f)
