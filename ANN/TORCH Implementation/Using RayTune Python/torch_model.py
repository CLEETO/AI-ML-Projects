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
import math

from functools import partial
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
class torch_model:
        

    def model_architure(self,layers):
        model=[]
        for i in range(len(layers)-2):
            model.append(nn.Linear(layers[i],layers[i+1]))
            model.append(nn.ReLU())     
        model.append(nn.Linear(layers[len(layers)-2],layers[len(layers)-1]))
        model.append(nn.Sigmoid())
        self.model=nn.Sequential(*model)
        return self.model
    


    def data_loader(self,batch_size):
        #absolute_path = os.path.abspath("dataset.csv")
        #dataset = pl.read_csv(absolute_path)
        dataset = pl.read_csv("Z:/My Folders/Cleeto_Tasks/task2/model_architecture_tweaked_copy/dataset.csv")
        exclude_col=[8,4]
        dataset=dataset.with_columns([pl.when(pl.col(column)==0).then(pl.col(column).mean()).otherwise(pl.col(column)) for id,column in enumerate(dataset.columns) if id not in exclude_col])
        scaler=RobustScaler()
        x=dataset[:,:8]
        y=dataset[:,8]
        x=scaler.fit_transform(x)
        x=torch.tensor(x,dtype=torch.float32)
        y=torch.tensor(y,dtype=torch.float32)
        dataset=TensorDataset(x,y)
        split_ratios=[int(math.floor(len(dataset)*0.8)),len(dataset)-int(math.floor(len(dataset)*0.8))]
        #print(len(dataset),split_ratios)
        generator=torch.Generator().manual_seed(50)
        train_data, val_data = random_split(dataset, split_ratios, generator=generator)
        train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
        val_dataloader=DataLoader(val_data,batch_size=16,shuffle=True)
        return train_dataloader,val_dataloader
    



    def train_model(self,config):
        #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #log_dir = os.path.join("runs", f"run_{timestamp}")
        #writer = SummaryWriter(log_dir=log_dir)
        model=self.model_architure(config['layers'])
        loss_fn = nn.BCELoss()  # binary cross entropy
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        epochs=100
        loss_total=0
        train_dataloader,val_dataloader=self.data_loader(config['batch_size'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for epoch in range(epochs):
            epoch_loss=0
            batch_count=0
            model.train()
            for x_train_batch,y_train_batch in train_dataloader:
                y_pred=model(x_train_batch)
                #print(x_train_batch,y_train_batch)
                y_pred = torch.squeeze(y_pred)
                y_train_batch=torch.squeeze(y_train_batch)
                loss=loss_fn(y_pred,y_train_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss+=loss.item()
                batch_count+=1
            #print("Epoch ",epoch,"loss: ",epoch_loss/batch_count)
            #writer.add_scalar('Train Loss',epoch_loss/batch_count,epoch)
            model.eval()
            val_loss=0
            with torch.no_grad():
                for x_val_batch, y_val_batch in val_dataloader:
                    y_pred = model(x_val_batch)
                    y_pred = torch.squeeze(y_pred)
                    loss = loss_fn(y_pred, y_val_batch)
                    val_loss += loss.item()
            val_loss /= len(val_dataloader)
            #writer.add_scalar('Validation Loss', val_loss, epoch)
            #print(f"Validation Loss: {val_loss}")
            checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            }
            #with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            #    checkpoint = None
                
            #    torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pth"))
                    
            #    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                # Send the current training result back to Tune
            train.report({"loss": val_loss})#, checkpoint=checkpoint)


        
        


