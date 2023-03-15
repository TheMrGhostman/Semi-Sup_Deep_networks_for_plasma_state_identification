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
from utils.utils import SWISH, get_activation, H_alpha_only, acc_tst
from utils.layers import DenseBlock, Flatten

parser = argparse.ArgumentParser()
parser.add_argument("--iloc", type=int, required=True, help="index in csv table with best models")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="size of each image batch")
parser.add_argument("--hidden", type=str, default="32-32", help="number of neurons in hidden layers")
parser.add_argument("--ydim", type=int, default=4, help="number of classes")
parser.add_argument("--q_zxy", type=str, default="False", help="type of encoder -> False -> q(z|x) or True -> q(z|x,y)")
parser.add_argument("--bnorm", type=int, default=0, help="use batch normalization (0 or 1) ") # when i had bool here i had problems
parser.add_argument("--activation", type=str, default="relu", help="activation function")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--scheduler", type=str, default=None, help="scheduler params \"milestone-gamma\" or \"milestone1-milestone2-gamma\" ")
parser.add_argument("--dropout", type=int, default=0, help="probability of dropout")
parser.add_argument("--sup_samples", type=float, default=1.0, help="ratio of labeled date")
parser.add_argument("--balanced", type=str, default="True", help="balanced classes within supervised samples")
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
	label_idx=149,
	seed=options.seed
)

print(data_loaders["sup"].dataset.X.mean()) # internal check for development purposes

xdim = [1,160] if options.halpha=="True" else list(data_loaders["unsup"].dataset.X.shape[1:])
print(f"xdim: {xdim}")

# LOAD ARCHITECTURE
df = pd.read_csv("results/vae-halpha.csv") if options.halpha=="True" else pd.read_csv("results/vae.csv")
df = df.sort_values(by=["best_loss"])
df_list = []
for vae_type in sorted(df["vae_type"].unique()): # alphabeticly "CVAE"-"RVAE-old"-"VAE"
    df_list.append(df[df["vae_type"] == vae_type].head(3)) # 9 total per all methods 
df= pd.concat(df_list)
tmp = df.iloc[options.iloc]["parameters"]

enc_idx=6

# VARIATIONAL AUTOENCODER   
# We are using previous experiments with VAEs and M1 classifiers as "model hyperparameters search"
# Therefore we choose architecture according to best results (VAE)
vae = recreate_model_from_params(
	df.iloc[enc_idx], #options.iloc
    xdim=xdim,
	file_folder=None, # file_folder=None => create new model without pretrained weights
	ydim=options.ydim, 
	q_zxy=options.q_zxy=="True"
)

# CLASSIFIER
# M1 based classifier for comparebility
hidden_str = copy.copy(options.hidden)
splited = options.hidden.split("-")
hidden =  [int(y) for y in splited]

tmp = recreate_model_from_params(
	df.iloc[options.iloc], 
    xdim=xdim,
	file_folder=None, # file_folder=None => create new model without pretrained weights
	ydim=0, 
	q_zxy=False
)

classifier = copy.deepcopy(tmp.encoder) # like core of M1_based classifiers
in_, out_ = classifier[-1].mu.in_features, classifier[-1].mu.out_features 

if hidden_str == "0":
	classifier[-1] = nn.Linear(in_features=in_, out_features=options.ydim) # out_ == zdim
else:
	classifier[-1] = nn.Linear(in_features=in_, out_features=out_) # out_ == zdim
	classifier = [classifier]
	hidden_neurons = [out_]+hidden
	for (i, o) in zip(hidden_neurons[:-1], hidden):
		classifier.append(
			DenseBlock(
				input_dim=i,
				output_dim=o,
				activation=get_activation(options.activation)(),
				batch_norm=bool(options.bnorm),
				dropout=options.dropout
			)
		)
	classifier.append(
		nn.Linear(in_features=hidden[-1], out_features=options.ydim)
	)
	classifier = nn.Sequential(*classifier)

# SEMI-SUPERVISED VAE #TODO flatten 
if options.q_zxy=="True":
	if isinstance(vae.encoder[0], Flatten): # this is VAE with dense layers
		encoder = q_zxy(
			dense=vae.encoder[1:],
			conv=vae.encoder[0]
		)
	else: # this is CVAE
		find_flatten_layer = np.array([isinstance(layer, Flatten) for layer in vae.encoder]) 
		# there is Flatten layer between convs and dense layers
		flatten_idx = np.argwhere(find_flatten_layer==True).item()
		encoder = q_zxy(
			dense=vae.encoder[flatten_idx+1:],
			conv=vae.encoder[:flatten_idx+1]
		)

else:
	encoder = q_zx(vae.encoder)

model = SSVAE(
	encoder= encoder,
	decoder= vae.decoder,
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

model_name = f"SSVAE-simple-hybrid_lr={options.lr}_bs={options.batch_size}_hidden={hidden_str}_bnorm={options.bnorm}_activation={options.activation}_dropout={options.dropout}_scheduler={options.scheduler}_encoder-idx={options.iloc}_q-zxy={options.q_zxy}_label-loc=150"
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
	early_stopping=300
)

print("everything prepared -> Starting training!")
lh = trainer(range(options.epochs), data_loaders["sup"], data_loaders["unsup"], data_loaders["val"])

trainer.loss_history["encoder_info"] = df.iloc[enc_idx].to_dict() #options.iloc
trainer.loss_history["clf_info"] = df.iloc[options.iloc].to_dict()
trainer.loss_history["seed"] = options.seed

# just for prediction
data_loaders = load_and_preprocess_new(mode="sup", transform=H_alpha_only() if options.halpha=="True" else None, batch_size=512,label_idx=149, validation=True)

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


