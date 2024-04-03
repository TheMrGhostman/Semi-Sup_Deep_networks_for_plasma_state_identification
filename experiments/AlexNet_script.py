# C:\Users\micha\AppData\Local\Programs\Python\Python310\python
import sys
sys.path.append("../")

import argparse
import copy
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 

from utils.inference import Trainer
from utils.datasets import load_and_preprocess, load_unseen_test_data
from utils.inception import Inception, InceptionBlock, correct_sizes
from utils.models import inception_time
from utils.utils import SWISH, get_activation, H_alpha_only, acc_tst

import utils.datasets as d
import wandb
import datetime
import sklearn

#1) Nastavit Arguenty
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")  #epoch=kolikart projedu cely dataset tim modelem, iterace*batch size=cely dataset, tj kolik iteraci v zavilosti na batch size je potreba udelat aby probehla jedna epoch?
parser.add_argument("--batch_size", type=int, default=128, help="size of each image batch")
parser.add_argument("--activation", type=str, default="relu", help="activation function")
parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
parser.add_argument("--scheduler", type=str, default="None", help="scheduler params \"milestone-gamma\" or \"milestone1-milestone2-gamma\" ")
parser.add_argument("--sup_samples", type=float, default=1.0, help="ratio of labeled date")
parser.add_argument("--balanced", type=str, default="True", help="balanced classes within supervised samples")
parser.add_argument("--seed", type=int, default=666, help="validation split seed")
parser.add_argument("--ui", type=int, default=np.random.randint(10000), help="unique identifier")

options = parser.parse_args()
print(options)

#2) Připravit data
dataloaders=d.load_and_preprocess("sup", batch_size=options.batch_size)

#3) vstupní velikost dat
xdim = list(dataloaders["sup"].dataset.X.shape[1:])
print(f"xdim: {xdim}") # 5, 160

#4) model
f = nn.Sequential(
    nn.Conv1d(xdim[0],96,11,stride=4), 
    nn.ReLU(),
    nn.MaxPool1d(3,stride=2),
    nn.Conv1d(96,256,5,padding=2),
    nn.ReLU(),
    nn.MaxPool1d(3,stride=2),
    nn.Conv1d(256,384,3,padding=1),
    nn.ReLU(),
    nn.Conv1d(384,384,3,padding=1),
    nn.ReLU(),
    nn.Conv1d(384,256,3,padding=1), 
    nn.ReLU(),
    nn.MaxPool1d(3,stride=2), #5x5x256
    nn.Flatten(), #udela z toho 1x6400
    nn.Linear(3*256,4096),# 3 jsme spočítali průchodem naprázdno
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096,4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096,1000),
    nn.Softmax(dim=1)
)

#x,y = next(iter(dataloaders["sup"]))
#print(f(x).shape)

#5) Optimizer
optimizer= torch.optim.Adam(f.parameters(), lr=options.lr)

#6) scheduler
if (options.scheduler != "None") and (options.scheduler != None):
	splited = options.scheduler.split("-")
	milestones = [int(y) for y in splited[:-1]]
	gamma = float(splited[-1])
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
else:
	scheduler = None

#7) model name
model_name = f"AlexNet_lr={options.lr}_bs={options.batch_size}_scheduler={options.scheduler}_ui={options.ui}"
#dt = datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S--")

run = wandb.init(
    # Set the project where this run will be logged
    project="bc-thesis",
    #name = f"{dt}{model_name}"
    # Track hyperparameters and run metadata
    config={
	"model": "AlexNet",
	"model_name": model_name,
        "learning_rate": options.lr,
        "epochs": options.epochs,
	"batch_size": options.batch_size,
	"activation": options.activation,
	"scheduler": options.scheduler,
	"sup_samples": options.sup_samples,
	"balanced": options.balanced,
	"seed": options.seed,
	"ui": options.ui
    },
)

#8) vytvořit trenéra
trener = Trainer(
		model=f,
		optimizer=optimizer,
		loss_function=nn.CrossEntropyLoss(),
		scheduler=scheduler,
		tensorboard=True,
		model_name=model_name,
		early_stopping=30000,
		save_path="../checkpoints",
		verbose=True
	)


#9) nechat trenéra naučit model
print("everything prepared -> Starting training!")
history = trener(epochs=range(options.epochs), train_loader=dataloaders["sup"], validation_loader=dataloaders["val"])

#10) 
trener.loss_history["seed"] = options.seed
trener.loss_history["options"] = options

test_discharges=load_unseen_test_data()

for key in test_discharges:
	results = {}
	y_hat, y = acc_tst(trener.model, test_discharges, key)
	if len(np.unique(y)) !=1:
		my_table = wandb.Table(columns=["labels", "predictions"], data=np.vstack([y, y_hat]).transpose())
		run.log({f"Predictions - {key}": my_table})
		wandb.config.update({f"Test/Accuracy-{key}": sklearn.metrics.accuracy_score(y,y_hat), f"Test/F1-{key}": sklearn.metrics.f1_score(y,y_hat, average="macro")})


torch.save(trener.loss_history, f"../saves/{model_name}.pt")

