import sys
import pathlib
import torch

from transformers import AutoTokenizer
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.distilbert_v0 import config
from components.models import FpModel
from components.loss_functions import MultiTaskLoss

###
# sample
###

# prepare
tokenizer = AutoTokenizer.from_pretrained(
    config["datamodule"]["dataset"]["base_model_name"],
    use_fast=config["datamodule"]["dataset"]["use_fast_tokenizer"]
)

# tokenize
token = tokenizer.encode_plus(
    "test",
    truncation=True,
    add_special_tokens=True,
    max_length=config["datamodule"]["dataset"]["max_length"],
    padding="max_length"
)
ids = torch.tensor([token["input_ids"]])
masks = torch.tensor([token["attention_mask"]])

# model
model = FpModel(config["model"])
pred = model(ids, masks)

pred = torch.unsqueeze(pred, dim=0)

# calc loss
# criterion = torch.nn.CrossEntropyLoss()
criterion = MultiTaskLoss("torch.nn.CrossEntropyLoss", 6)
xx = torch.tensor([[[0.8, 0.2, 0.0], [0.8, 0.2, 0.0]], [[0.8, 0.2, 0.0], [0.8, 0.2, 0.0]]]).float()
yy = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]).float()
criterion(xx, yy)

criterion(pred, labels)
