import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

class MinLoss:
    def __init__(self):
        self.value = np.nan

    def update(self, min_loss):
        self.value = np.nanmin([self.value, min_loss])

class ValidResult:
    def __init__(self):
        self.values = None

    def append(self, values):
        if self.values is None:
            self.values = values
            return self.values
        self.values = np.concatenate([self.values, values])
        return self.values

class ConfusionMatrix:
    def __init__(self, probs, labels, config):
        # const
        self.config = config
        self.probs = probs
        self.labels = labels

        # variables
        self.fig = plt.figure(figsize=[4, 4], tight_layout=True)

    def draw(self):
        idx_probs = np.argmax(self.probs, axis=1)
        idx_labels = np.argmax(self.labels, axis=1)

        df_confmat = pd.DataFrame(
            confusion_matrix(idx_probs, idx_labels),
            index=self.config["label"],
            columns=self.config["label"]
        )
        axis = self.fig.add_subplot(1, 1, 1)
        sns.heatmap(df_confmat, ax=axis, cmap="bwr", square=True, annot=True)
        axis.set_xlabel("label")
        axis.set_ylabel("pred")
        return self.fig

class F1Score:
    def __init__(self, probs, labels, config):
        # const
        self.config = config
        self.probs = probs
        self.labels = labels

        # variables
        self.f1_scores = {
            "macro": None,
            "micro": None
        }

    def calc(self):
        idx_probs = np.argmax(self.probs, axis=1)
        idx_labels = np.argmax(self.labels, axis=1)
        self.f1_scores = {
            "macro": f1_score(idx_probs, idx_labels, average="macro"),
            "micro": f1_score(idx_probs, idx_labels, average="micro")
        }
        return self.f1_scores

class LogLoss:
    def __init__(self, probs, labels, config):
        # const
        self.probs = probs
        self.labels = labels
        self.config = config
        self.prob_min = 10**(-15)
        self.prob_max = 1-10**(-15)

        # variables
        self.logloss = np.nan

    def calc(self):
        norm_probs = self.probs / np.sum(self.probs, axis=1)[:, None]
        log_probs = np.log(np.clip(norm_probs, self.prob_min, self.prob_max))
        self.logloss = -np.mean(np.sum(self.labels * log_probs, axis=1))
        return self.logloss

class MeanColumnwiseRootMeanSquareError:
    def __init__(self, probs, labels, config):
        # const
        self.config = config
        self.probs = probs
        self.labels = labels
        self.pred_scores, self.label_scores = self._calc_score()

    def _calc_score(self):
        pred_scores = np.array(self.config["label_val"])[
            np.argmax(self.probs, axis=2)
        ]
        label_scores = np.array(self.config["label_val"])[
            np.argmax(self.labels, axis=2)
        ]
        return pred_scores, label_scores

    def calc(self):
        error = self.pred_scores - self.label_scores
        mcrmse = np.mean(np.sqrt(np.mean(error**2, axis=1)))
        return mcrmse

class MultiTaskStats:
    def __init__(self, probs, labels, config):
        # const
        self.config = config
        self.label_num = len(self.config["labels"])
        self.probs = probs
        self.labels = labels
        self.pred_scores, self.label_scores = self._calc_score()

    def _calc_score(self):
        pred_scores = np.array(self.config["label_val"])[
            np.argmax(self.probs, axis=2)
        ]
        label_scores = np.array(self.config["label_val"])[
            np.argmax(self.labels, axis=2)
        ]
        return pred_scores, label_scores

    def calc(self):
        error = self.pred_scores - self.label_scores
        mean = np.mean(error, axis=0)
        std = np.std(error, axis=0)
        means = {label: mean[i] for i, label in enumerate(self.config["labels"])}
        stds = {label: std[i] for i, label in enumerate(self.config["labels"])}
        return means, stds

class ErrorHist:
    def __init__(self, probs, labels, config):
        # const
        self.config = config
        self.label_num = len(self.config["labels"])
        self.probs = probs
        self.labels = labels
        self.pred_scores, self.label_scores = self._calc_score()
        self.cmap = plt.get_cmap("tab10")

        # variables
        self.fig = plt.figure(figsize=[4, 2*self.label_num], tight_layout=True)
        self.axes = [
            self.fig.add_subplot(self.label_num, 1, i+1)
            for i in range(self.label_num)
        ]

    def _calc_score(self):
        pred_scores = np.array(self.config["label_val"])[
            np.argmax(self.probs, axis=2)
        ]
        label_scores = np.array(self.config["label_val"])[
            np.argmax(self.labels, axis=2)
        ]
        return pred_scores, label_scores

    def _calc_hist(self):
        error = self.pred_scores - self.label_scores
        bins = {}
        counts = {}
        for i, label in enumerate(self.config["labels"]):
            bin, count = np.unique(error[:, i], return_counts=True)
            bins[label] = bin
            counts[label] = count
        return bins, counts

    def draw(self):
        bins, counts = self._calc_hist()
        for i, label in enumerate(self.config["labels"]):
            self.axes[i]
            self.axes[i].bar(
                bins[label],
                counts[label],
                ec="black",
                width=0.5,
                color=self.cmap(i)
            )
            self.axes[i].set_ylabel(label)
            self.axes[i].set_axisbelow(True)
            self.axes[i].grid(axis="y")
        return self.fig

class ScatterPlot:
    def __init__(self, probs, labels, config):
        # const
        self.config = config
        self.label_num = len(self.config["labels"])
        self.probs = probs
        self.labels = labels
        self.pred_scores, self.label_scores = self._calc_score()
        self.cmap = plt.get_cmap("tab10")

        # variables
        self.fig = plt.figure(figsize=[9, 3*self.label_num], tight_layout=True)
        self.axes = [
            self.fig.add_subplot(int(np.floor(self.label_num/3)), 3, i+1)
            for i in range(self.label_num)
        ]

    def _calc_score(self):
        pred_scores = np.array(self.config["label_val"])[
            np.argmax(self.probs, axis=2)
        ]
        label_scores = np.array(self.config["label_val"])[
            np.argmax(self.labels, axis=2)
        ]
        return pred_scores, label_scores

    def draw(self):
        for i, label in enumerate(self.config["labels"]):
            self.axes[i].scatter(
                self.pred_scores[:, i],
                self.label_scores[:, i],
                color=self.cmap(i),
                alpha=0.01
            )
            self.axes[i].set_xlim([0.8, 5.2])
            self.axes[i].set_ylim([0.8, 5.2])
            self.axes[i].set_title(label)
            self.axes[i].set_xlabel("pred")
            self.axes[i].set_ylabel("label")
            self.axes[i].set_aspect("equal")
            self.axes[i].set_axisbelow(True)
            self.axes[i].grid()
        return self.fig
