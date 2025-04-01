import sys
import torch
import numpy as np
from torch.utils.data._utils.collate import default_convert
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset, random_split
from tqdm import tqdm
from colorama import Fore
import os
import random

from grapher import Grapher, IndexGraph
from utils import load_count_dict, load_cycle, negative_relation_sampling, load_text, load_quads, \
    myConvert, reshape_relation_prediction_ranking_data, element_wise_cos, cal_metrics, CosineEmbeddingLoss, \
    load_history_paths, reshape_ranking_data, reshape_ranking_data_rel
from torch.cuda.amp import GradScaler, autocast
from transformers import set_seed
from models import SentenceTransformer,TKG_Model
import torch.nn.functional as F
import argparse
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser(description='Relation Prediction')

parser.add_argument('--device', type=str, default='cuda:0',
                    help='CUDA device or CPU')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--dataset', type=str, default='FB15k-237-subset',
                    help='name of the dataset')
parser.add_argument('--path_dir', type=str, default=None,
                    help='location of extracted paths for each triplet')
parser.add_argument('--text_dir', type=str, default=None,
                    help='location of relation and entity texts')
parser.add_argument('--model_load_file', type=str, default=None,
                    help='location to load pretrained cycle model')
parser.add_argument('--model_save_dir', type=str, default=None,
                    help='location to save model')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--model', type=str, default='model',
                    help='sentence transformer model name on Hugging Face website (huggingface.co)')
parser.add_argument('--tokenizer', type=str, default='model',
                    help='tokenizer name on Hugging Face website (huggingface.co)')
parser.add_argument('--train_sample_num', type=int, default=-1,
                    help='number of training samples randomly sampled, use -1 for all data')
parser.add_argument('--valid_sample_num', type=int, default=-1,
                    help='number of validating samples randomly sampled, use -1 for all data')
parser.add_argument('--max_path_num', type=int, default=5,
                    help='number of paths loaded for each triplet')
parser.add_argument('--neg_sample_num_train', type=int, default=5,
                    help='number of negative training samples')
parser.add_argument('--neg_sample_num_valid', type=int, default=5,
                    help='number of negative validating samples')
parser.add_argument('--neg_sample_num_test', type=int, default=50,
                    help='number of negative testing samples')
parser.add_argument('--mode', type=str, default='tail',
                    help='whether head or tail is fixed')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--do_train',action='store_true', default=False,
                    help='whether train or not')
parser.add_argument('--do_test',action='store_true', default=False,
                    help='whether test or not')
parser.add_argument('--output_dir', type=str, default=None,
                    help='location to output test results')
args = parser.parse_args()


print(args)
set_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device(args.device)

if args.path_dir is None:
    path_dir = os.path.join("data/path_data/", args.dataset, f'ranking_{args.mode}/')
else:
    path_dir=args.path_dir

if args.text_dir is None:
    text_dir = os.path.join("data", args.dataset)
else:
    text_dir=args.text_dir


if args.model_load_file is None:
    model_load_file=os.path.join(f"save/{args.dataset}",f"relation_prediction_{args.mode}/best_val.pth")
else:
    model_load_file=args.model_load_file

if args.model_save_dir is None:
    model_save_dir=os.path.join(f"save/{args.dataset}", f"relation_prediction_{args.mode}/")
else:
    model_save_dir=args.model_save_dir
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
if args.output_dir is None:
    output_dir=os.path.join(f"output/{args.dataset}", f"relation_prediction_{args.mode}/")
else:
    output_dir=args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dataset = args.dataset
data_dir = "data/" + dataset + "/"
if dataset in ['WIKI','YAGO']:
    data = IndexGraph(data_dir)
else:
    data = Grapher(data_dir)

relation_dict = data.id2relation
entity_dict = data.id2entity
entity_dict[-1] = ''
time_dict = data.id2ts

train_quads,train_ts = load_quads(os.path.join(path_dir, "ranking_train.pickle"),relation_dict,entity_dict,time_dict)
train_paths = load_history_paths(os.path.join(path_dir, "paths_train.pickle"), len(train_quads),args.max_path_num,relation_dict,entity_dict,train_ts)
valid_quads, valid_ts = load_quads(os.path.join(path_dir, "ranking_valid.pickle"),relation_dict,entity_dict,time_dict)
valid_paths = load_history_paths(os.path.join(path_dir, "paths_valid.pickle"), len(valid_quads),args.max_path_num,relation_dict,entity_dict,valid_ts)
ranking_quads,test_ts = load_quads(os.path.join(path_dir, "ranking_test.pickle"),relation_dict,entity_dict,time_dict)
ranking_paths = load_history_paths(os.path.join(path_dir, "paths_test.pickle"), len(ranking_quads),args.max_path_num,relation_dict,entity_dict,test_ts)

train_triplets,train_paths,train_labels,_=reshape_ranking_data(train_quads[:,:-1],train_paths,args.neg_sample_num_train)
valid_triplets,valid_paths,valid_labels,_=reshape_ranking_data(valid_quads[:,:-1],valid_paths,args.neg_sample_num_valid)
ranking_triplets,ranking_paths,ranking_labels,ranking_indexes=reshape_ranking_data(ranking_quads[:,:-1],ranking_paths,args.neg_sample_num_test)

train_data = list(zip(train_triplets, train_paths,train_labels))
valid_data = list(zip(valid_triplets, valid_paths,valid_labels))
if args.train_sample_num==-1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler=RandomSampler(train_data,replacement=True,num_samples=args.train_sample_num)
train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size, collate_fn=myConvert)

if args.valid_sample_num==-1:
    valid_sampler = RandomSampler(valid_data)
else:
    valid_sampler = RandomSampler(valid_data,replacement=True,num_samples=args.valid_sample_num)
valid_data_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size, collate_fn=myConvert)

ranking_data = list(zip(ranking_triplets, ranking_paths,ranking_labels))
ranking_sampler = SequentialSampler(ranking_data)
ranking_data_loader = DataLoader(ranking_data, sampler=ranking_sampler, batch_size=args.batch_size, collate_fn=myConvert)

scaler = GradScaler()

model = TKG_Model(num_rels=len(relation_dict),device=0,h_dim=200)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
# 计算可训练参数数量（需梯度更新的参数）
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = torch.optim.AdamW(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)
scheduler=lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)
# criterion=torch.nn.CrossEntropyLoss()
criterion = CosineEmbeddingLoss

def train():
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        # ============================================ TRAINING ============================================================
        print(f"Training epoch {epoch}")
        training_pbar = tqdm(total=args.train_sample_num if args.train_sample_num>0 else len(train_data),
                             position=0, leave=True,
                             file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.RESET))
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for step, batch in enumerate(train_data_loader):
            sentence1, sentence2, targets = batch

            targets = torch.tensor(targets).to(device)
            optimizer.zero_grad()
            outputs = []
            with autocast():
                for s1, s2, tgt in zip(sentence1, sentence2, targets):
                    embed1 = model(*s1).unsqueeze(1)
                    embed2 = torch.stack([model(*s) for s in zip(*s2)])
                    sim = torch.cosine_similarity(embed1, embed2, dim=2)
                    outputs.append(sim)
                outputs = torch.stack(outputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()
            nb_tr_steps += 1
            training_pbar.update(len(targets))
        training_pbar.close()
        scheduler.step()
        print(f"Learning rate={optimizer.param_groups[0]['lr']}\nTraining loss={tr_loss / nb_tr_steps:.4f}")
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'{epoch}_best_val.pth'))
        if epoch % 3 == 0:
            valid_acc=validate()
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                torch.save(model.state_dict(), os.path.join(model_save_dir,'best_val.pth'))

    print(f"Best Validation Accuracy: {best_val_acc}")
def validate():
    valid_pbar = tqdm(total=args.valid_sample_num if args.valid_sample_num>0 else len(valid_data),
                     position=0, leave=True,
                     file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    model.eval()
    metrics = np.array([0., 0., 0., 0., 0.])  # MR, MRR, Hit@1, Hit@3, Hit@10
    nb_valid_steps = 0
    for batch in valid_data_loader:
        sentence1, sentence2, targets = batch
        targets = torch.tensor(targets).to(device)
        optimizer.zero_grad()
        outputs = []
        with autocast():
            with torch.no_grad():
                for s1, s2, tgt in zip(sentence1, sentence2, targets):
                    embed1 = model(*s1).unsqueeze(1)
                    embed2 = torch.stack([model(*s) for s in zip(*s2)])

                    sim,_ = torch.max(torch.cosine_similarity(embed1,embed2,dim=2),dim=1)
                    outputs.append(sim)
                outputs = torch.stack(outputs)
        metrics += cal_metrics(outputs.cpu().numpy(), targets.cpu().numpy())
        nb_valid_steps +=1
        valid_pbar.update(len(targets))
    valid_pbar.close()
    metrics = metrics / nb_valid_steps
    print(f"MR: {metrics[0]}, MRR: {metrics[1]}, Hit@1: {metrics[2]}, Hit@3: {metrics[3]}, Hit@10: {metrics[4]}")
    return metrics[2]
def test():
    ranking_pbar = tqdm(total=len(ranking_triplets),
                        position=0, leave=True,
                        file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
    model.eval()
    metrics=np.array([0.,0.,0.,0.,0.]) # MR, MRR, Hit@1, Hit@3, Hit@10
    nb_ranking_steps = 0
    scores=[]
    ranking_positions=[]
    for batch in ranking_data_loader:
        sentence1, sentence2, targets = batch

        targets = torch.tensor(targets).to(device)
        optimizer.zero_grad()
        outputs = []

        with autocast():
            with torch.no_grad():
                for s1, s2, tgt in zip(sentence1, sentence2, targets):
                    embed1 = model(*s1).unsqueeze(1)
                    embed2 = torch.stack([model(*s) for s in zip(*s2)])
                    sim,_ = torch.max(torch.cosine_similarity(embed1,embed2,dim=2),dim=1)

                outputs.append(sim)
                outputs = torch.stack(outputs)
        metrics+=cal_metrics(outputs.cpu().numpy(),targets.cpu().numpy())
        batch_scores = np.array(outputs.cpu().numpy())
        batch_positions = np.argsort(-batch_scores, axis=1)
        scores.append(batch_scores)
        ranking_positions.append(batch_positions)
        nb_ranking_steps += 1
        ranking_pbar.update(len(targets))

    scores=np.concatenate(scores)
    ordered_scores=np.array([scores[i][list(ranking_indexes[i])] for i in range(len(scores))])
    ordered_positions=np.argsort(-ordered_scores, axis=1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(os.path.join(output_dir,"indexes.txt"),ordered_positions,fmt="%d", delimiter="\t")
    np.savetxt(os.path.join(output_dir,"scores.txt"), ordered_scores,fmt="%.5f", delimiter="\t")
    metrics=metrics/nb_ranking_steps
    ranking_pbar.close()
    print(f"MR: {metrics[0]}, MRR: {metrics[1]}, Hit@1: {metrics[2]}, Hit@3: {metrics[3]}, Hit@10: {metrics[4]}")

if args.do_train:
    try:
        train()
    except KeyboardInterrupt:
        print("Receive keyboard interrupt, start testing:")
        model.load_state_dict(torch.load(model_load_file, map_location=device))
        test()
if args.do_test:
    model.load_state_dict(torch.load(model_load_file, map_location=device))
    test()