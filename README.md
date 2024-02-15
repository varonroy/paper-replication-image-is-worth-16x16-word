## Paper Replication - Image is Worth 16x16 Words
The goal of this project is not to replicate the paper one to one, but to take its core ideas and implement them manually.

- [paper](https://arxiv.org/abs/2010.11929)

## Data
The dataset is downloaded from `huggingface` and will be placed under `$HF_HOME/datasets`.

## Setup
required pakcages:
- `pytorch`
- `pytorch lightning`
- `matplotlib`
- `datasets`
- `torchmetrics`
- `torchinfo`

## Run
Show some samples from the dataset:
```bash
$ python data.py
```
Run a demo tensor through the model and print it's summary;
```bash
$ python model.py
```
Train the model:
```bash
$ python train.py
```
Run inference:
```bash
$ python train.py
```
Visualize the positional encoding maps:
```bash
$ python vis_pos.py
```
Visualize the activation maps
```
$ python vis_attention.py
```
