import sys
sys.path.append("../")

import argparse
import copy
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 

from utils.losses import Gaussian_NLL
from utils.ssvae import SSVAE, q_zxy, q_zx, SSVAE_Trainer
from utils.datasets import load_and_preprocess
from utils.inception import Inception, InceptionBlock, correct_sizes
from utils.models import inception_time, vae
from utils.utils import SWISH, get_activation, H_alpha_only, acc_tst

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="size of each image batch")
parser.add_argument("--zdim", type=int, default=15, help="dimension of latent z space")
parser.add_argument("--bneck", type=int, default=32, help="number of bottleneck channels")
parser.add_argument("--blocks", type=int, default=1, help="number of InceptionBlocks")
parser.add_argument("--n_filters", type=int, default=32, help="number of filters")
parser.add_argument("--fsizes", type=str, default="5-11-23", help="filters sizes of 3 conv layers in InceptionBlock ( 5-11-23)")
parser.add_argument("--activation", type=str, default="relu", help="activation function")
parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
parser.add_argument("--scheduler", type=str, default="None", help="scheduler params \"milestone-gamma\" or \"milestone1-milestone2-gamma\" ")
parser.add_argument("--es_trace", type=str, default="elbo_loss", help="criterion for early stopping: \"elbo_loss\" or \"clf_loss\" ")
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
	seed=options.seed
)
#label_idx=80

xdim = [1,160] if options.halpha=="True" else list(data_loaders["unsup"].dataset.X.shape[1:])
print(f"xdim: {xdim}")

# Variational Autoencoder
# Dense vae is better for training of ssvae with separate encoders q(y|x) & q(z|x) resp. q(z|x,y) 
VAE = vae(
    xdim=xdim, 
    zdim=options.zdim, 
    hidden_neurons=[128,128,128], 
    batch_norm=False, 
	activation=get_activation(options.activation), 
    dropout=False, 
    ydim=4, 
    q_zxy=True
)

# Classifier
# process options

splited = options.fsizes.split("-")
fsizes =  [int(y) for y in splited]
fsizes = correct_sizes(fsizes)
fsizes_str = '-'.join([str(y) for y in fsizes])

CLF = inception_time(
	xdim=xdim[0],
	ydim=4, 
	n_filters=options.n_filters, 
	fsizes=fsizes, 
	bottleneck=options.bneck, 
	blocks=options.blocks, 
	activation=options.activation
)

# Semi-Supervised VAE
encoder = q_zxy(
    dense=VAE.encoder[1:],
    conv=VAE.encoder[0] # Flatten layer is separate
)

model = SSVAE(
	encoder= encoder,
	decoder= VAE.decoder,
	classifier=CLF
)

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

model_name = f"SSVAE-InceptionTime_lr={options.lr}_bs={options.batch_size}_zdim={options.zdim}_bneck={options.bneck}_blocks={options.blocks}_nfilters={options.n_filters}_fsize={fsizes_str}_activation={options.activation}_scheduler={options.scheduler}"
if options.sup_samples != 1.0:
	balanced = "B" if options.balanced=="True" else "U"
	model_name += f"_sup-samples={options.sup_samples}-{balanced}"
if options.halpha=="True": 
	model_name += "_halpha"

#model_name +="_deriv=sgf"
#model_name += "_label-idx=160"

trainer = SSVAE_Trainer(
	model=model,
	optimizer=optimizer,
	loss_function=Gaussian_NLL(),
	alpha=alpha,
	scheduler=scheduler,
	model_name=model_name,
	save_path="checkpoints/",
	verbose=True,
	early_stopping=300,
    es_trace=options.es_trace
)

print("everything prepared -> Starting training!")
lh = trainer(range(options.epochs), data_loaders["sup"], data_loaders["unsup"], data_loaders["val"])

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
#torch.save(trainer.loss_history, "model_histories/label-idx/" + trainer.model_name + "_history.pt")
torch.save(trainer.loss_history, "model_histories/" + trainer.model_name + "_history.pt")