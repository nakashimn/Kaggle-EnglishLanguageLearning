import os
import random
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
import traceback

from components.preprocessor import DataPreprocessor, TextCleaner
from components.datamodule import FpDataset, DataModule
from components.models import FpModel
from config.deberta_v3_large_v0 import config

class Predictor:
    def __init__(
        self, Model, DataModule, Dataset, Tokenizer, df_test, config, transforms
    ):
        # const
        self.config = config
        self.df_test = df_test
        self.transforms = transforms

        # Class
        self.Model = Model
        self.DataModule = DataModule
        self.Dataset = Dataset
        self.Tokenizer = Tokenizer

        # variables
        self.results = None

    def run(self):
        # create datamodule
        datamodule = self._create_datamodule()

        # predict
        self.results = self._predict(datamodule)

        return self.results

    def _create_datamodule(self):
        datamodule = self.DataModule(
            df_train=None,
            df_val=None,
            df_pred=self.df_test,
            Dataset=self.Dataset,
            Tokenizer=self.Tokenizer,
            config=self.config["datamodule"],
            transforms=self.transforms
        )
        return datamodule

    def _predict(self, datamodule):
        # define trainer
        trainer = pl.Trainer(
            logger=None,
            **self.config["trainer"]
        )

        # load model
        model = self.Model.load_from_checkpoint(
            f"{self.config['path']['model_dir']}/{self.config['modelname']}.ckpt",
            config=self.config["model"],
            transforms=self.transforms
        )

        # prediction
        model.eval()
        with torch.inference_mode():
            results = trainer.predict(model, datamodule=datamodule)
        probs = np.concatenate([r["prob"].numpy() for r in results], axis=0)
        preds = np.argmax(probs, axis=2)
        return {"probs": probs, "preds": preds}

class PredictorEnsemble(Predictor):
    def _predict(self, datamodule):
        # define trainer
        trainer = pl.Trainer(
            logger=None,
            **self.config["trainer"]
        )

        probs_folds = []
        preds_folds = []
        for fold in range(self.config["n_splits"]):

            # load model
            model = self.Model.load_from_checkpoint(
                f"{self.config['path']['model_dir']}/{self.config['modelname']}_{fold}.ckpt",
                config=self.config["model"],
                transforms=self.transforms
            )

            # prediction
            model.eval()
            with torch.inference_mode():
                results = trainer.predict(model, datamodule=datamodule)
            probs = np.concatenate([r["prob"].numpy() for r in results], axis=0)
            probs_folds.append(probs)
            preds_folds.append(np.argmax(probs, axis=2))
        probs_ensemble = np.mean(probs_folds, axis=0)
        preds_ensemble = np.mean(preds_folds, axis=0)
        probs_folds = np.array(probs_folds)
        preds_folds = np.array(preds_folds)
        return {"probs": probs_ensemble, "preds": preds_ensemble,
            "probs_folds": probs_folds, "preds_folds": preds_folds}

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__=="__main__":

    fix_seed(config["random_seed"])

    # Setting Dataset
    data_preprocessor = DataPreprocessor(config, TextCleaner)
    df_test = data_preprocessor.test_dataset()

    # Prediction
    if config["pred_ensemble"]:
        cls_predictor = PredictorEnsemble
    else:
        cls_predictor = Predictor
    predictor = cls_predictor(
        FpModel,
        DataModule,
        FpDataset,
        AutoTokenizer,
        df_test,
        config,
        None
    )
    predictor.run()

    # output
    preds = np.mean(
        np.array(config["label_val"])[predictor.results["preds_folds"]], axis=0
    )
    submission = pd.concat([
        df_test[config["id"]],
        pd.DataFrame(preds, columns=config["labels"])
    ], axis=1)
    submission.to_csv("submission.csv", index=None)
