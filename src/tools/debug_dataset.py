import os
import random
import sys
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from transformers import AutoTokenizer
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.distilbert_v0 import config
from components.preprocessor import DataPreprocessor
from components.datamodule import FpDataset, DataModule

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###
# sample
###

data_preprocessor = DataPreprocessor(config)
df_train = data_preprocessor.train_dataset()

# FpDataSet
fix_seed(config["random_seed"])
dataset = FpDataset(df_train, config["datamodule"]["dataset"], AutoTokenizer)
batch = dataset.__getitem__(0)
ids = torch.unsqueeze(batch[0], 0)
masks = torch.unsqueeze(batch[1], 0)
labels = torch.unsqueeze(batch[2], 0)

# DataModule
data_module = DataModule(df_train, None, None, FpDataset, AutoTokenizer, config["datamodule"], None)
for batch in data_module.train_dataloader():
    ids = batch[0]
    masks = batch[1]
    labels = batch[2]
    break
