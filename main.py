import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning import LightningModule, LightningDataModule

import matplotlib.pyplot as plt

# config
BATCH_SIZE = 4096
TRAINING_RESOLUTION = 224
LEARNING_RATE_WARMUP_STEPS = 10000
