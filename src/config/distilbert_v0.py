config = {
    "n_splits": 5,
    "train_fold": [0, 1, 2, 3, 4],
    "valid_fold": [0, 1, 2, 3, 4],
    "random_seed": 57,
    "id": "text_id",
    "group": "text_id",
    "features": [
        "full_text"
    ],
    "labels": [
        "cohesion",
        "syntax",
        "vocabulary",
        "phraseology",
        "grammar",
        "conventions"
    ],
    "label_val": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "experiment_name": "fp-el-distilbert-v0",
    "path": {
        "traindata": "/kaggle/input/feedback-prize-english-language-learning/train.csv",
        "testdata": "/kaggle/input/feedback-prize-english-language-learning/test.csv",
        "temporal_dir": "../tmp/artifacts/",
        "model_dir": "/kaggle/input/fp-el-distilbert-v0/"
    },
    "modelname": "best_loss",
    "pred_ensemble": True,
    "train_with_alldata": False
}
config["model"] = {
    "base_model_name": "/kaggle/input/distilbertbaseuncased/",
    "dim_feature": 768,
    "num_class": 9,
    "dropout_rate": 0.5,
    "labels": config["labels"],
    "freeze_base_model": False,
    "enable_gradient_checkpoint": False,
    "loss": {
        "name": "MultiTaskLoss",
        "params": {
            "LossFunction": "nn.CrossEntropyLoss",
            "num_task": 6,
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
    },
    "optimizer":{
        "name": "optim.RAdam",
        "params":{
            "lr": 1e-5
        },
    },
    "scheduler":{
        "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params":{
            "T_0": 20,
            "eta_min": 1e-4,
        }
    }
}
config["earlystopping"] = {
    "patience": 1
}
config["checkpoint"] = {
    "dirpath": config["path"]["temporal_dir"],
    "monitor": "val_loss",
    "save_top_k": 1,
    "mode": "min",
    "save_last": False,
    "save_weights_only": False
}
config["trainer"] = {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": 100,
    "accumulate_grad_batches": 1,
    "fast_dev_run": False,
    "deterministic": True,
    "num_sanity_val_steps": 0,
    "resume_from_checkpoint": None,
    "precision": 16
}
config["kfold"] = {
    "name": "GroupKFold",
    "params": {
        "n_splits": config["n_splits"]
    }
}
config["datamodule"] = {
    "dataset":{
        "base_model_name": config["model"]["base_model_name"],
        "num_class": config["model"]["num_class"],
        "features": config["features"],
        "labels": config["labels"],
        "label_val": config["label_val"],
        "use_fast_tokenizer": True,
        "additional_special_tokens": None,
        "max_length": 512
    },
    "train_loader": {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": False
    },
    "pred_loader": {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": False,
        "drop_last": False
    }
}
config["Metrics"] = {
    "labels": config["labels"],
    "label_val": config["label_val"]
}
