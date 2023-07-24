from . import datasets, model_utils, cpd_models, klcpd, tscp

import os

import torch
from torch.utils.data import Subset

from abc import ABC
from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

EPS = 1e-6

class EnsembleCPDModel(ABC):
    """Wrapper for general ensemble models with bootstrapping."""

    def __init__(
        self,
        args: dict,
        n_models: int,
        boot_sample_size: int = None,
        seed: int = 0,
        train_anomaly_num: int = None
    ) -> None:
        """Initialize EnsembleCPDModel.

        :param args: dictionary containing core model params, learning params, loss params, etc.
        :param n_models: number of models to train
        :param boot_sample_size: size of the bootstrapped train dataset 
                                 (if None, all the models are trained on the original train dataset)
        :param seed: random seed to be fixed
        """
        super().__init__()
                
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
            experiments_name=args["experiments_name"],
            train_anomaly_num=train_anomaly_num
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

            cpd_models.fix_seeds(i)

            curr_model = model_utils.get_models_list(
                self.args,
                self.train_datasets_list[i],
                self.test_dataset
            )[-1] # list consists of 1 model as, currently, we do not work with 'combined' models
            self.models_list.append(curr_model)

    def fit(self, monitor: str = "val_loss", patience: int = 10, min_delta: float = 0.0) -> None:
        """Fit all the models on the corresponding train datasets.
        
        :params monitor, patience: Early Stopping parameters
        """
        logger = TensorBoardLogger(save_dir=f'logs/{self.args["experiments_name"]}', name=self.args["model_type"])
        
        if not self.fitted:
            self.initialize_models_list()
            for i, (cpd_model, train_dataset) in enumerate(zip(self.models_list, self.train_datasets_list)):

                cpd_models.fix_seeds(i)
                
                print(f'\nFitting model number {i + 1}.')
                trainer = pl.Trainer(
                    max_epochs=self.args["learning"]["epochs"],
                    gpus=self.args["learning"]["gpus"],
                    #accelerator=self.args["learning"]["accelerator"],
                    #devices=self.args["learning"]["devices"],
                    benchmark=True,
                    check_val_every_n_epoch=1,
                    logger=logger,
                    callbacks=EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience)
                )
                trainer.fit(cpd_model)
            
            self.fitted = True
            
        else:
            print("Attention! Models are already fitted!")

    def predict(self, inputs: torch.Tensor, scale: int = None) -> torch.Tensor:
        """Make a prediction.
        
        :param inputs: input batch of sequences
        :param scale: scale parameter for KL-CPD and TSCP2 models
        
        :returns: torch.Tensor containing predictions of all the models
        """
        
        if not self.fitted:
            print("Attention! The model is not fitted yet.")
            
        ensemble_preds = []
        for model in self.models_list:
            if self.args["model_type"] == "seq2seq":
                ensemble_preds.append(model(inputs))
            elif self.args["model_type"] == "kl_cpd":
                outs = klcpd.get_klcpd_output_scaled(model, inputs, model.window_1, model.window_2, scale=scale)
                ensemble_preds.append(outs)
            elif self.args["model_type"] == "tscp":
                #outs = tscp.get_tscp_output_scaled(model, inputs, model.window_1, model.window_2, scale=scale)
                outs = tscp.get_tscp_output_scaled_padded(model, inputs, model.window_1, model.window_2, scale=scale)
                ensemble_preds.append(outs)
            else:
                raise ValueError(f'Wrong or not implemented model type {self.args["model_type"]}.')
        
        # shape is (n_models, batch_size, seq_len)
        ensemble_preds = torch.stack(ensemble_preds)
        n_models, batch_size, seq_len = ensemble_preds.shape

        preds_mean = torch.mean(ensemble_preds, axis=0).reshape(batch_size, seq_len)
        preds_std = torch.std(ensemble_preds, axis=0).reshape(batch_size, seq_len)
        
        # store current predictions
        self.preds = ensemble_preds

        return preds_mean, preds_std
    
    def get_quantile_predictions(self, inputs: torch.Tensor, q: float, scale: int = None) -> torch.Tensor:
        """Get the q-th quantile of the predicted CP scores distribution.

        :param inputs: input batch of sequences
        :param q: desired quantile
        :param scale: scale parameter for KL-CPD and TSCP2 models

        :returns: torch.Tensor containing quantile predictions
        """
        _, preds_std = self.predict(inputs)
        preds_quantile = torch.quantile(self.preds, q, axis=0)
        return preds_quantile, preds_std

    def get_min_predictions(self, inputs: torch.Tensor, scale: int = None) -> torch.Tensor:
        """Get the point-wise minimum of the predicted CP scores distribution.

        :param inputs: input batch of sequences
        :param scale: scale parameter for KL-CPD and TSCP2 models

        :returns: torch.Tensor containing quantile predictions
        """
        _, preds_std = self.predict(inputs)
        
        # torch.min() returs a tuple (values, indices)
        preds_min = torch.min(self.preds, axis=0)[0]
        return preds_min, preds_std

    def get_max_predictions(self, inputs: torch.Tensor, scale: int = None) -> torch.Tensor:
        """Get the point-wise maximum of the predicted CP scores distribution.

        :param inputs: input batch of sequences
        :param scale: scale parameter for KL-CPD and TSCP2 models

        :returns: torch.Tensor containing quantile predictions
        """
        _, preds_std = self.predict(inputs)

        # torch.max() returs a tuple (values, indices)
        preds_max = torch.max(self.preds, axis=0)[0]
        return preds_max, preds_std

    def save_models_list(self, path_to_folder: str) -> None:
        """Save trained models.
        
        :param path_to_folder: path to the folder for saving, e.g. 'saved_models/mnist' 
        """
        
        if not self.fitted:
            print("Attention! The models are not trained.")
        
        loss_type = self.args["loss_type"] if self.args["model_type"] == "seq2seq" else None
        
        for i, model in enumerate(self.models_list):
            path = (
                path_to_folder + "/" +
                self.args["experiments_name"] + "_loss_type_" +
                str(loss_type) + "_model_type_" +             
                self.args["model_type"] + "_sample_" +
                str(self.boot_sample_size) + "_model_num_" + str(i) + ".pth"
            )
            torch.save(model.state_dict(), path)
            
    def load_models_list(self, path_to_folder: str) -> None:
        """Load weights of the saved models from the ensemble.

        :param path_to_folder: path to the folder for saving, e.g. 'saved_models/mnist'
        """
        # check that the folder contains self.n_models files with models' weights,
        # ignore utility files
        paths_list = [path for path in os.listdir(path_to_folder) if not path.startswith('.')]
                
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
        seed: int = 0,
        train_anomaly_num: int = None,
        cusum_mode: str = "correct"
    ) -> None:
        """Initialize EnsembleCPDModel.

        :param args: dictionary containing core model params, learning params, loss params, etc.
        :param n_models: number of models to train
        :param boot_sample_size: size of the bootstrapped train dataset 
                                 (if None, all the models are trained on the original train dataset)
        :param scale_by_std: if True, scale the statistic by predicted std, i.e.
                                in cusum, t = series_mean[i] - series_mean[i-1]) / series_std[i],
                             else:
                                t = series_mean[i] - series_mean[i-1]
        :param susum_threshold: threshold for CUSUM algorithm
        :param seed: random seed to be fixed
        """
        super().__init__(args, n_models, boot_sample_size, seed, train_anomaly_num)

        self.cusum_threshold = cusum_threshold
        self.scale_by_std = scale_by_std
        
        assert cusum_mode in ["correct", "old"], "Wrong CUSUM mode"
        self.cusum_mode = cusum_mode
    
    def cusum_detector_batch(
        self, series_batch: torch.Tensor, series_std_batch: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CUSUM change point detection.
        
        :param series_mean:
        :param series_std:
        
        :returns: change_mask 
        """
        batch_size, seq_len = series_batch.shape
        
        normal_to_change_stat = torch.zeros(batch_size, seq_len).to(series_batch.device)
        change_mask = torch.zeros(batch_size, seq_len).to(series_batch.device)
        
        for i in range(1, seq_len):
            # old CUSUM
            if self.cusum_mode == "old":
                if self.scale_by_std:
                    t = (series_batch[:, i] - series_batch[:, i - 1]) / (series_std_batch[:, i] + EPS)

                else:
                    t = series_batch[:, i] - series_batch[:, i - 1]
                    
            # new (correct) CUSUM
            else:
                t = (series_batch[:, i] - 0.5) / (series_std_batch[:, i] ** 2 + EPS)

            normal_to_change_stat[:, i] = torch.maximum(
                torch.zeros(batch_size).to(series_batch.device),
                normal_to_change_stat[:, i - 1] + t
            )
            
            is_change = normal_to_change_stat[:, i] > torch.ones(batch_size).to(series_batch.device) * self.cusum_threshold
            change_mask[is_change, i:] = True
            
        return change_mask, normal_to_change_stat
    
    def sample_cusum_trajectories(self, inputs):
                
        if not self.fitted:
            print("Attention! The model is not fitted yet.")
            
        ensemble_preds = []
        
        for model in self.models_list:
            ensemble_preds.append(model(inputs).squeeze())
        
        # shape is (n_models, batch_size, seq_len)
        ensemble_preds = torch.stack(ensemble_preds)
        
        _, batch_size, seq_len = ensemble_preds.shape
        
        preds_mean = torch.mean(ensemble_preds, axis=0).reshape(batch_size, seq_len)
        preds_std = torch.std(ensemble_preds, axis=0).reshape(batch_size, seq_len)
        
        cusum_trajectories = []
        change_masks = []
        
        for preds_traj in ensemble_preds:
            # use one_like tensor of std's, do not take them into account
            #change_mask, normal_to_change_stat = self.cusum_detector_batch(preds_traj, torch.ones_like(preds_traj))
            change_mask, normal_to_change_stat = self.cusum_detector_batch(preds_traj, preds_std)
            cusum_trajectories.append(normal_to_change_stat)
            change_masks.append(change_mask)
        
        cusum_trajectories = torch.stack(cusum_trajectories)
        change_masks = torch.stack(change_masks)
        
        return change_masks, cusum_trajectories
    
    def predict(self, inputs: torch.Tensor, scale: int = None) -> torch.Tensor:
        """Make a prediction.
        
        :param inputs: input batch of sequences
        
        :returns: torch.Tensor containing predictions of all the models
        """
        
        if not self.fitted:
            print("Attention! The model is not fitted yet.")
            
        ensemble_preds = []
        for model in self.models_list: 
            if self.args["model_type"] == "seq2seq":
                ensemble_preds.append(model(inputs))
            elif self.args["model_type"] == "kl_cpd":
                outs = klcpd.get_klcpd_output_scaled(model, inputs, model.window_1, model.window_2, scale=scale)
                ensemble_preds.append(outs)
            elif self.args["model_type"] == "tscp":
                #outs = tscp.get_tscp_output_scaled(model, inputs, model.window_1, model.window_2, scale=scale)
                outs = tscp.get_tscp_output_scaled_padded(model, inputs, model.window_1, model.window_2, scale=scale)
                ensemble_preds.append(outs)
            else:
                raise ValueError(f'Wrong or not implemented model type {self.args["model_type"]}.')
        
        # shape is (n_models, batch_size, seq_len)
        ensemble_preds = torch.stack(ensemble_preds) 
        n_models, batch_size, seq_len = ensemble_preds.shape

        preds_mean = torch.mean(ensemble_preds, axis=0).reshape(batch_size, seq_len)
        preds_std = torch.std(ensemble_preds, axis=0).reshape(batch_size, seq_len)
        
        # store current predictions
        self.preds = ensemble_preds
        
        change_masks, normal_to_change_stats = self.cusum_detector_batch(preds_mean, preds_std)

        self.preds_mean = preds_mean
        self.preds_std = preds_std
        self.change_masks = change_masks
        self.normal_to_change_stats = normal_to_change_stats
         
        return change_masks
    
    def predict_cusum_trajectories(self, inputs: torch.Tensor, q: float = 0.5) -> torch.Tensor:
        """Make a prediction.
        
        :param inputs: input batch of sequences
        
        :returns: torch.Tensor containing predictions of all the models
        """
        change_masks, _ = self.sample_cusum_trajectories(inputs)
        
        cp_idxs_batch = torch.argmax(change_masks, dim=2).float()
        
        cp_idxs_batch_aggr = torch.quantile(cp_idxs_batch, q, axis=0).round().int()
        
        _, bs, seq_len = change_masks.shape
        
        cusum_quantile_labels = torch.zeros(bs, seq_len).to(inputs.device)
        
        for b in range(bs):
            if cp_idxs_batch_aggr[b] > 0:
                cusum_quantile_labels[b, cp_idxs_batch_aggr[b]:] = 1
        
        return cusum_quantile_labels