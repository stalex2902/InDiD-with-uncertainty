from . import datasets, model_utils, cpd_models

import numpy as np
import os

import torch
from torch.utils.data import Subset

from abc import ABC

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class EnsembleCPDModel(ABC):
    """Wrapper for general ensemble models with bootstrapping."""

    def __init__(
        self,
        args: dict,
        n_models: int,
        boot_sample_size: int=None,
        seed: int = 0
    ) -> None:
        """Initialize EnsembleCPDModel.

        :param args: dictionary containing core model params, learning params, loss params, etc.
        :param n_models: number of models to train
        :param boot_sample_size: size of the bootstrapped train dataset 
                                 (if None, all the models are trained on the original train dataset)
        """
        super().__init__()
        
        # fix seed for reproducibility
        cpd_models.fix_seeds(seed)
        
        self.args = args
        
        assert args["experiments_name"] in [
            "synthetic_1D", 
            "synthetic_100D",
            "mnist",
            "human_activity",
            "explosion",
            "road_accidents"
        ], "Wrong experiments name"
                
        self.train_dataset, self.test_dataset = datasets.CPDDatasets(
            experiments_name=args["experiments_name"]
        ).get_dataset_()
        
        self.n_models = n_models
        
        if boot_sample_size is not None:
            assert boot_sample_size <= len(self.train_dataset), "Desired sample size is larger than the whole train dataset."
        self.boot_sample_size = boot_sample_size
        
        self.fitted = False

        self.initialize_models_list()

    def eval(self) -> None:
        """Turn all the models to 'eval' mode (for consistency with our code)."""
        for model in self.models_list:
            model.eval()
    
    def to(self, device: str) -> None:
        """Move all models to the device (for consistency with our code)."""
        for model in self.models_list:
            model.to(device) 
        
    def bootstrap_datasets(self) -> None:
        """Generate new train datasets if necessary."""
        # No boostrap
        if self.boot_sample_size is None:
            self.train_datasets_list = [self.train_dataset] * self.n_models
            
        else:
            self.train_datasets_list = []
            for _ in range(self.n_models):
                
                # sample with replacement
                idxs = np.random.choice(range(len(self.train_dataset)), size=self.boot_sample_size)
                curr_train_data = Subset(self.train_dataset, idxs)
                self.train_datasets_list.append(curr_train_data)

    def initialize_models_list(self) -> None:
        """Initialize cpd models for a particular exeriment."""
        self.bootstrap_datasets()

        self.models_list = []
        for i in range(self.n_models):
            curr_model = model_utils.get_models_list(
                self.args,
                self.train_datasets_list[i],
                self.test_dataset
            )[-1] # list consists of 1 model as, currently, we do not work with 'combined' models
            self.models_list.append(curr_model)

    def fit(self) -> None:
        """Fit all the models on the corresponding train datasets."""
        if not self.fitted:
            self.initialize_models_list()
            for i, (cpd_model, train_dataset) in enumerate(zip(self.models_list, self.train_datasets_list)):
                print(f'Fitting model number {i+1}.')
                trainer = pl.Trainer(
                    max_epochs=self.args["learning"]["epochs"],
                    accelerator=self.args["learning"]["accelerator"],
                    devices=self.args["learning"]["devices"],
                    benchmark=True,
                    check_val_every_n_epoch=1,
                    callbacks=EarlyStopping(monitor="val_loss", min_delta=0, patience=10)
                )
                trainer.fit(cpd_model)
            
            self.fitted = True
            
        else:
            print("Attention! Models are already fitted!")

    def predict(self, inputs) -> torch.Tensor:
        """Make a prediction.
        
        :param inputs: input batch of sequences
        
        :returns: torch.Tensor containing predictions of all the models
        """
        
        if not self.fitted:
            print("Attention! The model is not fitted yet.")
            
        ensemble_preds = []
        
        for model in self.models_list:
            ensemble_preds.append(model(inputs))
        
        # shape is (n_models, batch_size, seq_len)
        ensemble_preds = torch.stack(ensemble_preds)

        preds_mean = torch.mean(ensemble_preds, axis=0)
        preds_std = torch.std(ensemble_preds, axis=0) 
        
        # store current predictions
        self.preds = ensemble_preds
        
        return preds_mean, preds_std

    def save_models_list(self, path_to_folder) -> None:
        """Save trained models."""
        
        if not self.fitted:
            print("Attention! The models are not trained.")
        
        for i, model in enumerate(self.models_list):
            path = (
                path_to_folder + "/" +
                self.args["experiments_name"] + "_" +
                self.args["loss_type"] + "_sample_" + 
                str(self.boot_sample_size) + "_model_num_" + str(i) + ".pth"
            )
            torch.save(model.state_dict(), path)
            
    def load_models_list(self, path_to_folder) -> None:
        paths_list = os.listdir(path_to_folder)
        assert len(paths_list) == self.n_models, "Number of paths is not equal to the number of models."
        
        # initialize models list
        self.initialize_models_list()
        
        # load state dicts
        for model, path in zip(self.models_list, paths_list):
            model.load_state_dict(torch.load(path_to_folder + "/" + path))
        
        self.fitted = True