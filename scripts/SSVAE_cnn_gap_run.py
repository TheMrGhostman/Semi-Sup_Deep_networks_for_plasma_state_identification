import sys
sys.path.append("../")

import argparse
import copy
import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F 

from utils.datasets import load_and_preprocess
from utils.losses import Gaussian_NLL
from utils.ssvae import SSVAE, q_zxy, q_zx, SSVAE_Trainer
from utils.load_utils import recreate_model_from_params, parse_parameters
from utils.utils import get_activation, H_alpha_only, acc_tst
from utils.layers import Flatten, VariationalLayer, VariationalDecoderOutput
from utils.models import cvae2

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="size of each image batch")
parser.add_argument("--nfilters", type=str, default="64-64-64", help="number of filters for convolutional layers")
parser.add_argument("--zdim", type=int, default=15, help="latent dimensions")
parser.add_argument("--ydim", type=int, default=4, help="number of classes")
parser.add_argument("--activation", type=str, default="relu", help="activation function")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--scheduler", type=str, default=None, help="scheduler params \"milestone-gamma\" or \"milestone1-milestone2-gamma\" ")
parser.add_argument("--sup_samples", type=float, default=1.0, help="ratio of labeled date")
parser.add_argument("--balanced", type=str, default="True", help="balanced classes within supervised samples")
parser.add_argument("--es_trace", type=str, default="elbo_loss", help="criterion for early stopping: \"elbo_loss\" or \"clf_loss\" ")
parser.add_argument("--halpha", type=str, default="False", help="use only h_alpha")
parser.add_argument("--seed", type=int, default=666, help="validation split seed")

options = parser.parse_args()
print(options)

# DATA
data_loaders = load_and_preprocess(
	mode="semisup", 
	transform=H_alpha_only() if options.halpha=="True" else None,
	batch_size=options.batch_size, 
	validation=True, 
	one_hot=True,
	sub_samples=options.sup_samples,
	balanced=options.balanced=="True", 
	seed=options.seed
)

print(data_loaders["sup"].dataset.X.mean()) # internal check for development purposes

xdim = [1,160] if options.halpha=="True" else list(data_loaders["unsup"].dataset.X.shape[1:])
print(f"xdim: {xdim}")


n_filters = options.nfilters.split("-")
n_filters = [int(y) for y in n_filters]

if len(n_filters)==3:
	config = {
		"fsizes": [6,4,4],
		"stride": [2,2,2],
		"padding":[0,0,0],
		"pooling":[0,0,0]
	}
elif len(n_filters)==4:
	config = {
		"fsizes": [5,3,3,3],
		"stride": [1,1,1,1],
		"padding":[2,1,1,1],
		"pooling":[0,2,0,2]
	}
else: 
	raise ValueError("Depth of convolutional part is not 3 or 4")

# CLASSIFIER
classifier = cvae2(
	xdim=xdim,
	zdim=1, # not needed
	n_filters=copy.copy(n_filters),
	fsizes=config["fsizes"],
	stride=config["stride"],
	padding=config["padding"],
	pooling=config["pooling"],
	dense=None,
	activation=get_activation(options.activation)
).encoder[:-2] # cut dense and flatten

print(n_filters[-1], "filter -1")

classifier = nn.Sequential(
	*classifier,
	nn.AdaptiveAvgPool1d(output_size=1),
	Flatten(out_features=n_filters[-1]),
	nn.Linear(in_features=n_filters[-1], out_features=options.ydim)
)

# VARIATIONAL AUTOENCODER
encoder = q_zxy(
	dense= nn.Sequential(
		nn.Linear(in_features=xdim[0]*xdim[1] + options.ydim, out_features=128),
		get_activation(options.activation)(),
		nn.Linear(in_features=128, out_features=128),
		get_activation(options.activation)(),
		nn.Linear(in_features=128, out_features=128),
		get_activation(options.activation)(),
		VariationalLayer(in_features=128, out_features=options.zdim, return_KL=False)
	),
	conv=Flatten(out_features=xdim[0]*xdim[1])
)

decoder = nn.Sequential(
	nn.Linear(in_features=options.zdim + options.ydim, out_features=128),
	get_activation(options.activation)(),
	nn.Linear(in_features=128, out_features=128),
	get_activation(options.activation)(),
	nn.Linear(in_features=128, out_features=128),
	get_activation(options.activation)(),
	VariationalDecoderOutput(
		in_features=128, 
		mu_out=xdim[0]*xdim[1], 
		sigma_out=xdim[0],  
		reshape=True
	)
)

model = SSVAE(
	encoder= encoder,
	decoder= decoder,
	classifier=classifier
)

print(model)

alpha = 0.1*data_loaders["sup"].dataset.X.size(0) # from KINGMA paper

# OPTIMIZER and SCHEDULER
optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)

if (options.scheduler != None) and (options.scheduler != "None"):
	splited = options.scheduler.split("-")
	milestones = [int(y) for y in splited[:-1]]
	gamma = float(splited[-1])
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
else:
	scheduler = None

model_name = f"SSVAE-GAP+VAE-simple_lr={options.lr}_bs={options.batch_size}_nfilters={options.nfilters}_zdim={options.zdim}_activation={options.activation}_scheduler={options.scheduler}"
if options.sup_samples != 1.0:
	balanced = "B" if options.balanced=="True" else "U"
	model_name += f"_sup-samples={options.sup_samples}-{balanced}"
if options.halpha=="True": 
	model_name += "_halpha"

trainer = SSVAE_Trainer(
	model=model,
	optimizer=optimizer,
	loss_function=Gaussian_NLL(),
	alpha=alpha,
	scheduler=scheduler,
	model_name=model_name,
	save_path="checkpoints/",#"results/temporary/",#
	verbose=True,
	es_trace=options.es_trace,
	early_stopping=300
)

print("everything prepared -> Starting training!")
lh = trainer(range(options.epochs), data_loaders["sup"], data_loaders["unsup"], data_loaders["val"])

trainer.loss_history["encoder_info"] = {"vae_type":"Dense", "hidden":"128-128-128"}
trainer.loss_history["clf_info"] = config
trainer.loss_history["seed"] = options.seed
trainer.loss_history["options"] = options

# just for prediction
data_loaders = load_and_preprocess_new(mode="sup", transform=H_alpha_only() if options.halpha=="True" else None, batch_size=512, validation=True)

predictor = copy.deepcopy(trainer.model)
predictor.load_state_dict(trainer.early_stopping.best_model)
predictor = predictor.cpu()

y_hat, y = acc_tst(predictor.classifier, data_loaders, split="test")
trainer.loss_history["y_pred_test"] = y_hat
trainer.loss_history["y_test"] = y

y_hat, y = acc_tst(predictor.classifier, data_loaders, split="val")
trainer.loss_history["y_pred_val"] = y_hat
trainer.loss_history["y_val"] = y


#torch.save(trainer.loss_history, "/results/temporary/" + trainer.model_name + "_history.pt")
torch.save(trainer.loss_history, "model_histories/" + trainer.model_name + "_history.pt")


