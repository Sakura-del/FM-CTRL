import pickle
import sys
import torch
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset, random_split
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from colorama import Fore
import os
import random
from collections import defaultdict
import copy
import torch.nn.functional as F


def load_count_dict(path):
    count_dict = {}
    with open(os.path.join(path, "relation_cycle_count.txt"), encoding='utf-8') as f:
        for line in f:
            v, k = line.split(": ")
            k = k.rstrip('\t\n')
            k = " ".join(sorted(k.split("\t")))
            count_dict[k] = float(v)
    return count_dict


def load_triplets(path):
    triplets = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            h, r, t = line.split()
            triplets.append([h, r, t])
    return triplets

def load_quads(path,relation_dict,entity_dict,time_dict):
    with open(path,'rb') as f:
        quads = pickle.load(f)
        quads = np.array(quads)
        ts = quads[:,3]
        rels = quads[:,1]
        s = [entity_dict[i] for i in quads[:,0]]
        r = [relation_dict[i] for i in quads[:,1]]
        o = [entity_dict[i]  for i in quads[:,2]]
        t = [time_dict[i] for i in quads[:,3]]
        quads = np.stack((s,r,o,t),axis=1)
    return quads,ts

def load_text(path):
    dict = {'entity': {}, 'relation': {}}
    relation_texts = []
    with open(os.path.join(path, "entity2text.txt"), encoding='utf-8') as f:
        for line in f:
            entity, text = line.split("\t")
            text = text.rstrip('\n')
            dict['entity'][entity] = text

    with open(os.path.join(path, "relation2text.txt"), encoding='utf-8') as f:
        for line in f:
            relation, text = line.split("\t")
            text = text.rstrip('\n')
            words = text.split()
            dict['relation'][relation] = " ".join(words)
            relation_texts.append(" ".join(words))
            words.reverse()
            dict['relation']["{" + relation + "}^-1"] = " ".join(words)
            relation_texts.append(" ".join(words))
    dict['relation']['nopath'] = ""
    return dict, relation_texts

def inverse_relation(r):
    if r[0] == "{" and r[-4:] == "}^-1":
        return r[1:-4]
    else:
        return "{" + r + "}^-1"


triple2id = {'relation': {'2id': {}, 'id2': []}, 'entity': {'2id': {}, 'id2': []}}


def tokenize(dataset):
    tokens = {'relation': [], 'entity': []}
    for line in dataset['entity']:
        token_line = []
        for (index, word) in enumerate(line):
            if word not in triple2id['entity']['2id']:
                triple2id['entity']['2id'][word] = len(triple2id['entity']['id2'])
                triple2id['entity']['id2'].append(word)
            token_line.append(triple2id['entity']['2id'][word])
        tokens['entity'].append(token_line)
    for line in dataset['relation']:
        token_line = []
        for (index, word) in enumerate(line):
            if word not in triple2id['relation']['2id']:
                triple2id['relation']['2id'][word] = len(triple2id['relation']['id2'])
                triple2id['relation']['id2'].append(word)
            token_line.append(triple2id['relation']['2id'][word])
        tokens['relation'].append(token_line)
    # for (index, label) in enumerate(dataset['label']):
    #     if label not in triple2id['relation']['2id']:
    #         triple2id['relation']['2id'][label]=len(triple2id['relation']['id2'])
    #         triple2id['relation']['id2'].append(label)
    #     tokens['label'].append(triple2id['relation']['2id'][label])
    return tokens


def load_neg_relation(path):
    f = open(os.path.join(path, "negative_relations.txt"), encoding="utf-8")
    neg_relations = []
    for line in f:
        r = line.rstrip('\t\n').split("\t")
        neg_relations.append(r)
    return neg_relations


max_seq_length = 128


def load_paths(relation_dir, entity_dir, data_size, max_path_num):
    paths = []
    f1 = open(relation_dir, encoding='utf-8')
    f2 = open(entity_dir, encoding='utf-8')
    for i in range(data_size):
        pnum1 = int(f1.readline())
        pnum2 = int(f2.readline())
        assert pnum1 == pnum2
        paths.append([])
        for j in range(pnum1):
            relations = f1.readline().rstrip('\t\n').split("\t")
            entities = f2.readline().rstrip('\t\n').split("\t")
            if j >= max_path_num:
                continue

            p = [rv for r in zip(entities, relations) for rv in r]
            p.append(entities[-1])
            paths[-1].append(p)
        for i in range(pnum1, max_path_num):
            paths[-1].append(['nopath'])
    return paths

# source function
def load_history_paths(path_file, data_size, max_path_num,relation_dict,entity_dict,tq):
    text_paths = []
    time_stamps = []
    path_nums = []
    with open(path_file,'rb') as f:
        load_paths = pickle.load(f)
    for i in tqdm(range(data_size)):
        paths = load_paths[i]
        pnum = len(paths)
        text_paths.append([])
        time_stamps.append([])
        path_nums.append(pnum)
        for path in paths:
            relations = [relation_dict[j] for j in path[1::4]]
            time_span = [tq[i] - j for j in path[3::4]]
            time_stamps[-1].append(time_span)
            entities = [entity_dict[j] for j in path[0::4]]
            # if j >= max_path_num:
            #     continue

            p = [rv for r in zip(entities, relations) for rv in r]
            p.append(entity_dict[path[-2]])
            text_paths[-1].append(p)
            # text_paths[-1].append(relations)
        for i in range(pnum, max_path_num):
            text_paths[-1].append([""])
            time_stamps[-1].append([-1])
    return list(zip(text_paths,time_stamps))

def reshape_ranking_data(ranking_triplets, ranking_paths, neg_size):
    reshaped_triplets = []
    reshaped_paths = []
    labels = []
    indexes = []
    path_ts = []
    path_nums = []
    path_rels = []
    for i in range(0, len(ranking_triplets), neg_size):
        triplets = ranking_triplets[i:i + neg_size].tolist()
        paths,times = zip(*ranking_paths[i:i + neg_size])
        inverse_index = list(np.arange(neg_size))
        truth = triplets[0]
        t_p = list(zip(triplets, paths,times, inverse_index))
        random.shuffle(t_p)
        triplets, paths,times, inverse_index = zip(*t_p)
        label = triplets.index(truth)
        reshaped_triplets.append(triplets)
        reshaped_paths.append(paths)
        labels.append(label)
        index = np.argsort(inverse_index)
        indexes.append(index)
        path_ts.append(times)
    reshaped_triplets = [["; ".join([er for er in st]) + " [SEP]" for st in st1] for st1 in reshaped_triplets]
    t_q = [[[0]] * neg_size] * len(reshaped_triplets)
    reshaped_paths = [[["; ".join([er for er in s]) + " [SEP]" for s in st] for st in st2] for st2 in
                      reshaped_paths]
    return list(zip(reshaped_triplets,t_q)), list(zip(reshaped_paths,path_ts)), labels,indexes


def myConvert(data):
    for i in range(len(data[0])):
        yield [d[i] for d in data]


def element_wise_cos(a, b):
    a_norm = torch.sqrt(torch.sum(torch.square(a), dim=1)).unsqueeze(0)
    b_norm = torch.sqrt(torch.sum(torch.square(b), dim=1)).unsqueeze(0)
    return torch.matmul(b, a.T) / torch.matmul(b_norm.T, a_norm)


def cal_metrics(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    sorted_index = np.argsort(-predictions, axis=1)
    pos = np.array([np.where(sorted_index[i] == labels[i]) for i in range(len(labels))]).flatten()
    hit1 = len(pos[pos < 1]) / len(pos)
    hit3 = len(pos[pos < 3]) / len(pos)
    hit10 = len(pos[pos < 10]) / len(pos)
    MR = np.average(pos + 1)
    MRR = np.average(1 / (pos + 1))
    return [MR, MRR, hit1, hit3, hit10]


def CosineEmbeddingLoss(outputs, labels, margin=0.):
    scores = outputs.permute(2, 0, 1)
    y = F.one_hot(labels, num_classes=outputs.shape[1]).repeat(outputs.shape[2], 1, 1)
    pos_loss = (1 - scores) * y
    neg_loss = (1 - y) * torch.clamp(scores - margin, min=0)
    loss = torch.sum(pos_loss + neg_loss) / outputs.shape[2]
    return loss


def store_history_graph(quads):
    """
    Store all neighbors (outgoing edges) for each node.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        neighbors (dict): neighbors for each node
    """

    # history_graph = defaultdict(lambda: defaultdict(list))
    neighbors = dict()
    nodes = list(set(quads[:, 0]))
    for node in nodes:
        neighbors[node] = quads[quads[:, 0] == node]
        # for quad in neighbors[node]:
        #     history_graph[node][quad[2]].append(quad[[1, 3]])

    return neighbors

def split_by_time_with_t(data, time_list):
    snapshot_with_t_list = []
    for t in time_list:
        snapshot_with_t = data[data[:, 3] == t]
        snapshot_with_t_list.append(snapshot_with_t)
    return snapshot_with_t_list
