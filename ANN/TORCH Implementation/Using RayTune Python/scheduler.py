import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
from sklearn.preprocessing import RobustScaler
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import ray
from torch_model import *
from ray.train import RunConfig
class scheduler:

   
    def generate_layers(self):
            layers=[8]
            num_of_hidden_layers=np.random.randint(1,8)
            for i in range (num_of_hidden_layers):
                layers.append(np.random.randint(6,36))
            layers.append(1)
            return layers

    def main(self):
            config={'layers':tune.choice([self.generate_layers() for _ in range(10)]),"lr": tune.loguniform(1e-4, 1e-1), "batch_size": tune.choice([16,32,64])}
            #config={'layers':[8,16,12,8,4,1],"lr": tune.loguniform(1e-4, 1e-1), "batch_size": tune.choice([16,32])}
            
            scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=100,
            grace_period=40,
            reduction_factor=2,
            )   
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join("logs", f"run_{timestamp}")  # Shorten the path
            writer = SummaryWriter(log_dir=log_dir)
            result=tune.Tuner(torch_model().train_model,
            param_space=config,
            tune_config=tune.TuneConfig(
            num_samples=200,
            scheduler=scheduler),
            run_config=RunConfig(
        name="experiment_name",
        storage_path="~/ray_results/",
    )
            
            
            
            )
            result=result.fit()
            best_result = result.get_best_result("loss", "min")
            print("Best trial config: {}".format(best_result.config))
            print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
            
if __name__=='__main__':
      scheduler().main()