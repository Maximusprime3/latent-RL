import os
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
load_pretrained = True
if load_pretrained:
    ckpt_path='logs/BCE_sum_VAE/MSSIMVAE//version_23/checkpoints/last.ckpt'
    ckpt = torch.load(ckpt_path)
    experiment = VAEXperiment(model, config['exp_params'])
    experiment.load_state_dict(ckpt['state_dict'])
else:
    experiment = VAEXperiment(model,
                          config['exp_params'])
# init the weights with HE init, seed?
# def init_weights(m):
#    if isinstance(m, nn.Linear):
#        torch.nn.init.he_uniform(m.weight)
#        m.bias.data.fill_(0.01)
#    elif isinstance(m, nn.Conv2d):
#        torch.nn.init.he_uniform(m.weight)
#        m.bias.data.fill_(0.01)  
#    elif isinstance(m, nn.ConvTranspose2d):
#        torch.nn.init.he_uniform(m.weight)
#        m.bias.data.fill_(0.01)
# model.apply(init_weights)



data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)