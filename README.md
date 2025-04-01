# FM-CTRL

This is the code for the paper "Coherent Temporal Logical Reasoning via Fusing Multi-faced Information for Link Forecasting over Temporal Knowledge Graphs".

## Requirements

- python 3.6.0
- colorama 0.4.4
- matplotlib 3.3.4
- networkx 2.5.1
- numpy 1.18.5
- scikit_learn 1.1.2
- torch 1.9.1
- tqdm 4.64.0
- transformers 4.18.0

## Quick Start

We provide an all-in-one file `generate_train_and_test.sh` to automatically extract paths, train model, and test model. Before you begin, please prepare a "model" folder containing pre-trained language models such as BERT, along with their configuration files.

## Datesets

The three datasets we proposed in the paper can all be found in the "data" folder.