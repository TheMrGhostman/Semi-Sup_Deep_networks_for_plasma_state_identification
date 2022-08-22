import sys
sys.path.append("../")

import argparse
import copy
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 

from utils.inference import Trainer
from utils.datasets import load_and_preprocess_new
from utils.inception import Inception, InceptionBlock, correct_sizes
from utils.models import inception_time
from utils.utils import SWISH, get_activation, H_alpha_only, acc_tst

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="size of each image batch")
parser.add_argument("--bneck", type=int, default=32, help="number of bottleneck channels")
parser.add_argument("--blocks", type=int, default=1, help="number of InceptionBlocks")
parser.add_argument("--n_filters", type=int, default=32, help="number of filters")
parser.add_argument("--fsizes", type=str, default="5-11-23", help="filters sizes of 3 conv layers in InceptionBlock ( 5-11-23)")
parser.add_argument("--activation", type=str, default="relu", help="activation function")
parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
parser.add_argument("--scheduler", type=str, default="None", help="scheduler params \"milestone-gamma\" or \"milestone1-milestone2-gamma\" ")
parser.add_argument("--sup_samples", type=float, default=1.0, help="ratio of labeled date")
parser.add_argument("--balanced", type=str, default="True", help="balanced classes within supervised samples")
parser.add_argument("--halpha", type=str, default="False", help="use only h_alpha")
parser.add_argument("--seed", type=int, default=666, help="validation split seed")

options = parser.parse_args()
print(options)

data_loaders = load_and_preprocess_new(
	mode="sup", 
	transform=H_alpha_only() if options.halpha=="True" else None,
	batch_size=options.batch_size, 
	validation=True, 
	sub_samples=options.sup_samples, 
	balanced=options.balanced=="True",
	seed=options.seed
)

print(data_loaders["sup"].dataset.X.mean()) # internal check for development purposes

xdim = [1,160] if options.halpha=="True" else list(data_loaders["sup"].dataset.X.shape[1:])
print(f"xdim: {xdim}")

splited = options.fsizes.split("-")
fsizes =  [int(y) for y in splited]
fsizes = correct_sizes(fsizes)
fsizes_str = '-'.join([str(y) for y in fsizes])

IT = inception_time(
	xdim=xdim[0],
	ydim=4, 
	n_filters=options.n_filters, 
	fsizes=fsizes, 
	bottleneck=options.bneck, 
	blocks=options.blocks, 
	activation=options.activation
) #max params => 300740 weights :D

optimizer= torch.optim.Adam(IT.parameters(), lr=options.lr)

if (options.scheduler != "None") and (options.scheduler != None):
	splited = options.scheduler.split("-")
	milestones = [int(y) for y in splited[:-1]]
	gamma = float(splited[-1])
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
else:
	scheduler = None

model_name = f"InceptionTime_lr={options.lr}_bs={options.batch_size}_bneck={options.bneck}_blocks={options.blocks}_nfilters={options.n_filters}_fsize={fsizes_str}_scheduler={options.scheduler}"
if options.sup_samples != 1.0:
	balanced = "B" if options.balanced=="True" else "U"
	model_name += f"_sup-samples={options.sup_samples}-{balanced}"
if options.halpha=="True": 
	model_name += "_halpha"

model_name +="_deriv=sgf"

m1 = Trainer(
		model=IT,
		optimizer=optimizer,
		loss_function=nn.CrossEntropyLoss(),
		scheduler=scheduler,
		tensorboard=True,
		model_name=model_name,
		early_stopping=30,
		save_path="checkpoints/",
		verbose=True
	)

print("everything prepared -> Starting training!")
lh = m1(epochs=range(options.epochs), train_loader=data_loaders["sup"], validation_loader=data_loaders["val"])

m1.loss_history["seed"] = options.seed

predictor = copy.deepcopy(m1.model)
predictor.load_state_dict(m1.early_stopping.best_model)
predictor = predictor.cpu()

y_hat, y = acc_tst(predictor, data_loaders, split="test")
m1.loss_history["y_pred_test"] = y_hat
m1.loss_history["y_test"] = y

y_hat, y = acc_tst(predictor, data_loaders, split="val")
m1.loss_history["y_pred_val"] = y_hat
m1.loss_history["y_val"] = y

torch.save(m1.loss_history, "model_histories/" + m1.model_name + "_history.pt")