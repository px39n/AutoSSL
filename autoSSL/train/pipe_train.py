import os
import glob
import yaml
import pytorch_lightning as pl
from autoSSL.utils import ck_callback, join_dir,ContinuousCSVLogger
import time
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
class Trainer(pl.Trainer):
    def __init__(self, config, model_mode, extra_epoch=0,early_stop=False, **kwargs):
        self.config = config.copy()
        self.model_mode = model_mode
        self.extra_epoch = extra_epoch
        self.early_stop=early_stop 
        # Define the path for the config and checkpoint
        self.config["log_dir"] = join_dir(config["checkpoint_dir"], config["experiment"], config["name"])

        # Create directory if not exists
        os.makedirs(self.config["log_dir"], exist_ok=True)

        # Initialize or load config depending on the mode
        if self.model_mode == "start":
            self.save_config()
        elif self.model_mode == "continue":
            self.load_config()
            self.save_config()
            self.update_max_epoch()
           
        if self.early_stop==True:
            callbacks=[EarlyStopping(monitor="kNN_accuracy", mode="max", patience=3)]
        else:
            callbacks=[]
        super().__init__(
            max_epochs=self.config["max_epochs"],
            accelerator=self.config["device"],
            callbacks=callbacks+[ck_callback(self.config["log_dir"])],
            logger=ContinuousCSVLogger(save_dir=self.config["log_dir"]),
            **kwargs 
        )

    def save_config(self):
        # Save the config
        with open(join_dir(self.config["log_dir"], "config.yaml"), 'w') as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

    def load_config(self):
        # Load the config
        with open(join_dir(self.config["log_dir"], "config.yaml"), 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        
    def update_max_epoch(self):    
        # Update max_epochs and save updated config
        print(f"Max Epoch: {self.config['max_epochs']+self.extra_epoch}=\
        Previous epoch({self.config['max_epochs']})+Extra epoch({self.extra_epoch})")
        self.config["max_epochs"] += self.extra_epoch

    def extract_epoch(self,strs):        
        epoch_str = strs.split('=')[1].split('-')[0]
        return int(epoch_str)    
            
    def fit(self, model, dataloader, val_dataloaders=None, ckpt_path=None ):
        if self.model_mode == "continue" and ckpt_path == "latest":
            # Get the most recent checkpoint file based on the epoch number in the filename
            checkpoint_files = glob.glob(join_dir(self.config["log_dir"], "*.ckpt"))
            max_epoch = -1
            for file in checkpoint_files:
                # extract epoch number from filename
                epoch=self.extract_epoch(file)
                if epoch > max_epoch:
                    max_epoch = epoch
                    ckpt_path = file
            print(f"Loading checkpoint from: {ckpt_path}, current epoch: {epoch}")
            
            
            
        start = time.time()
        
        super().fit(model, dataloader,val_dataloaders=model.dataloader_kNN, ckpt_path=ckpt_path)
        
        end = time.time()
        
        if "runing time(min)" in self.config:
            self.config["runing time(min)"] += (end-start)
        else:
            self.config["runing time(min)"] = (end-start)/ 60
        self.config["GPU Usage(Gbyte)"] = torch.cuda.max_memory_allocated() / (1024**3)    
        torch.cuda.reset_peak_memory_stats()
        self.save_config() 
        