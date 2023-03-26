from . import datasets, model_utils, cpd_models

import os

import torch
from torch.utils.data import Subset

from abc import ABC
from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class EnsembleCPDModel(ABC):
    """Wrapper for general ensemble models with bootstrapping."""

    def __init__(
        self,
        args: dict,
        n_models: int,
        boot_sample_size: int = None,
        seed: int = 0
    ) -> None:
        """Initialize EnsembleCPDModel.

        :param args: dictionary containing core model params, learning params, loss params, etc.
        :param n_models: number of models to train
        :param boot_sample_size: size of the bootstrapped train dataset 
                                 (if None, all the models are trained on the original train dataset)
        :param seed: random seed to be fixed
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
                idxs = torch.randint(len(self.train_dataset), size=(self.boot_sample_size, ))
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
                print(f'\nFitting model number {i+1}.')
                trainer = pl.Trainer(
                    max_epochs=self.args["learning"]["epochs"],
                    gpus=self.args["learning"]["gpus"],
                    benchmark=True,
                    check_val_every_n_epoch=1,
                    callbacks=EarlyStopping(monitor="val_loss", min_delta=0, patience=10)
                )
                trainer.fit(cpd_model)
            
            self.fitted = True
            
        else:
            print("Attention! Models are already fitted!")

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
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
    
    def get_quantile_predictions(self, inputs: torch.Tensor, q: float) -> torch.Tensor:
        """Get the q-th quantile of the predicted CP scores distribution.

        :param inputs: input batch of sequences
        :param q: desired quantile

        :returns: torch.Tensor containing quantile predictions
        """
        _, preds_std = self.predict(inputs)
        preds_quantile = torch.quantile(self.preds, q, axis=0)
        return preds_quantile, preds_std

    def save_models_list(self, path_to_folder: str) -> None:
        """Save trained models.
        
        :param path_to_folder: path to the folder for saving, e.g. 'saved_models/mnist' 
        """
        
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
            
    def load_models_list(self, path_to_folder: str) -> None:
        """Load weights of the saved models from the ensemble.

        :param path_to_folder: path to the folder for saving, e.g. 'saved_models/mnist'
        """
        # check that the folder contains self.n_models files with models' weights
        paths_list = os.listdir(path_to_folder)
        assert len(paths_list) == self.n_models, "Number of paths is not equal to the number of models."
        
        # initialize models list
        self.initialize_models_list()
        
        # load state dicts
        for model, path in zip(self.models_list, paths_list):
            model.load_state_dict(torch.load(path_to_folder + "/" + path))
                    
        self.fitted = True


class CusumEnsembleCPDModel(EnsembleCPDModel):
    """Wrapper for cusum aproach ensemble models."""

    def __init__(
        self,
        args: dict,
        n_models: int,
        boot_sample_size: int = None,
        scale_by_std: bool = True,
        cusum_threshold: float = 0.1,
        seed: int = 0
    ) -> None:
        """Initialize EnsembleCPDModel.

        :param args: dictionary containing core model params, learning params, loss params, etc.
        :param n_models: number of models to train
        :param boot_sample_size: size of the bootstrapped train dataset 
                                 (if None, all the models are trained on the original train dataset)
        :param scale_by_std: if True, scale the statistic by predicted std, i.e.
                                in cusum, t = series_mean[i] - series_mean[i-1]) / series_std[i] * self.std_scaler,
                             else:
                                t = series_mean[i] - series_mean[i-1]
        :param susum_threshold: threshold for CUSUM algorithm
        :param seed: random seed to be fixed
        """
        super().__init__(args, n_models, boot_sample_size, seed)
        self.cusum_threshold = cusum_threshold
        self.scale_by_std = scale_by_std

    def cusum_detector(
        self, series_mean: torch.Tensor, series_std: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CUSUM change point detection.
        
        :param series_mean:
        :param series_std:
        
        :returns: change_mask 
        """
        normal_to_change_stat = torch.zeros(len(series_mean)).to(series_mean.device)
        #change_to_normal_stat = torch.zeros(len(series_mean)).to(series_mean.device)
        change_mask = torch.zeros(len(series_mean)).to(series_mean.device)

        for i in range(1, len(series_mean)):

            if self.scale_by_std:
                t = (series_mean[i] - series_mean[i-1]) / series_std[i]
            
            else:
                t = series_mean[i] - series_mean[i-1]

            normal_to_change_stat[i] = max(0, normal_to_change_stat[i - 1] + t)
            
            if normal_to_change_stat[i] > self.cusum_threshold:
                change_mask[i:] = True
                break          
            
        return change_mask, normal_to_change_stat
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
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

        change_masks = []
        normal_to_change_stats = []
        for pred_mean, pred_std in zip(preds_mean, preds_std):
            change_mask, normal_to_change_stat = self.cusum_detector(pred_mean, pred_std)
            change_masks.append(change_mask)
            normal_to_change_stats.append(normal_to_change_stat)

        change_masks = torch.stack(change_masks)
        normal_to_change_stats = torch.stack(normal_to_change_stats)

        self.preds_mean = preds_mean
        self.preds_std = preds_std
        self.change_masks = change_masks
        self.normal_to_change_stats = normal_to_change_stats
        
        return change_masks