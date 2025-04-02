# FM-CTRL

This is the code for the paper "Coherent Temporal Logical Reasoning via Fusing Multi-faced Information for Link Forecasting over Temporal Knowledge Graphs".

## Requirements

- python 3.9.6
- colorama 0.4.6
- matplotlib 3.8.3
- networkx 3.2.1
- numpy 1.26.4
- torch 2.2.2
- tqdm 4.66.2
- transformers 4.39.3
- peft 0.10.0

## Quick Start

We provide an all-in-one file `generate_train_and_test.sh` to automatically extract paths, train model, and test model. Before you begin, please prepare a "model" folder containing pre-trained language models such as BERT, along with their configuration files.

## Datesets

The three datasets we proposed in the paper can all be found in the "data" folder.