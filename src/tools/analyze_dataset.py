import os
import sys
import argparse
import pathlib
import glob
import datetime
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from text_unidecode import unidecode
import traceback

sys.path.append(str(pathlib.Path(__file__).parents[1]))

from config.distilbert_v0 import config

if __name__=="__main__":

    filepath_train = "/kaggle/input/feedback-prize-english-language-learning/train.csv"
    filepath_test = "/kaggle/input/feedback-prize-english-language-learning/test.csv"

    df_train = pd.read_csv(filepath_train)

    # header
    headers_score = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

    # resolution of score
    np.unique(df_train[headers_score])
    ### 1.0 ~ 5.0 , 0.5刻み 9クラス

    # histogram
    bins = np.arange(1.0, 5.01, 0.5)
    plt.hist(df_train["cohesion"], bins=bins)
    plt.hist(df_train["syntax"], bins=bins)
    plt.hist(df_train["vocabulary"], bins=bins)
    plt.hist(df_train["phraseology"], bins=bins)
    plt.hist(df_train["grammar"], bins=bins)
    plt.hist(df_train["conventions"], bins=bins)
    plt.hist(df_train[headers_score].mean(axis=1), bins=np.arange(1.0, 5.01, 0.5/6))

    # corrcoef
    sns.heatmap(df_train.corr(), cmap="bwr", square=True, annot=True)

    # text
    tokenizer = AutoTokenizer.from_pretrained(
        "/kaggle/input/distilbertbaseuncased",
        use_fast=True
    )
    token_length = []
    for idx in tqdm(df_train.index):
        text = df_train.loc[idx, "full_text"]
        token = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=2560,
            padding="max_length"
        )
        ids = torch.tensor([token["input_ids"]])
        masks = torch.tensor([token["attention_mask"]])
        token_length.append(int(masks.sum()))

    plt.hist(token_length)

    np.max(token_length)            # 1435
    np.quantile(token_length, 0.99) # 1238
    np.quantile(token_length, 0.95) # 890
    np.median(token_length)         # 460
    np.quantile(token_length, 0.05) # 216
    np.quantile(token_length, 0.01) # 141
    np.min(token_length)            # 16

    df_train["token_length"] = token_length
    mean_score = df_train[headers_score].mean(axis=1).values
    np.corrcoef(token_length, mean_score)  # 0.20
    plt.scatter(token_length, mean_score, alpha=0.3)

    sns.heatmap(df_train.corr(), cmap="bwr", square=True, annot=True)

    # onehot-encoding
    labels = [
        torch.concat([F.one_hot(
                torch.tensor([config["label_val"].index(df_train.loc[idx, label])]),
                num_classes=config["model"]["num_class"]
            ).float() for label in config["labels"]
        ]) for idx in df_train.index
    ]

    #
    df_train[["full_text"]].values.flatten()


    df_train[["full_text"]]
    text = df_train.loc[0, "full_text"]

    unidecode(text.encode("raw_unicode_escape").decode("utf-8").encode("cp1252").decode("utf-8"))
