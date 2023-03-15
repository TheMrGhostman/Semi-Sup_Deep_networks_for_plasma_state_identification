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
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="size of each image batch")
parser.add_argument("--ydim", type=int, default=4, help="number of classes")
parser.add_argument("--q_zxy", type=str, default="False", help="type of encoder -> False -> q(z|x) or True -> q(z|x,y)")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--scheduler", type=str, default=None, help="scheduler params \"milestone-gamma\" or \"milestone1-milestone2-gamma\" ")
parser.add_argument("--es_trace", type=str, default="clf_loss", help="criterion for early stopping: \"elbo_loss\" or \"clf_loss\" ")
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

print(data_loaders["sup"].dataset.X.mean()) # internal check for development purposes

xdim = [1,160] if options.halpha=="True" else list(data_loaders["unsup"].dataset.X.shape[1:])
print(f"xdim: {xdim}")

# LOAD ARCHITECTURE
df = pd.read_csv("results/vae-halpha.csv") if options.halpha=="True" else pd.read_csv("results/vae.csv")
df = df.sort_values(by=["best_loss"])
df_list = []
for vae_type in sorted(df["vae_type"].unique()): # alphabeticly "CVAE"-"RVAE-old"-"VAE"
    df_list.append(df[df["vae_type"] == vae_type].head(5)) # 9 total per all methods 
df= pd.concat(df_list)

vae_idx=10 # VAE

# VARIATIONAL AUTOENCODER   
# We are using previous experiments with VAEs and M1 classifiers as "model hyperparameters search"
# Therefore we choose architecture according to best results (VAE)
vae = recreate_model_from_params(
	df.iloc[vae_idx],
    xdim=xdim,
	file_folder=None, # file_folder=None => create new model without pretrained weights
	ydim=options.ydim, 
	q_zxy=options.q_zxy=="True"
)

# CLASSIFIER
# M1 based classifier for comparebility
class CRNN(nn.Module):
    def __init__(self, in_channels=5, device=torch.device("cuda")):
        super(CRNN, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(in_channels,64,3,padding=1), # 5
            nn.ReLU(),
            nn.Conv1d(64,128,3,padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2),
            nn.Conv1d(128,256,3,padding=1),
            nn.ReLU(),
            nn.Conv1d(256,256,3,padding=1),
            nn.ReLU(),
            nn.Conv1d(256,256,3,padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2),
            nn.Conv1d(256,256,3,padding=1),
            nn.ReLU(),
            nn.Conv1d(256,256,3,padding=1),
            nn.ReLU(),
            nn.Conv1d(256,256,3,padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(5*256,64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(32, 32, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(32,32),
            nn.Dropout(0.5),
            nn.Linear(32,4)
        )
        self.device=device

    def forward(self, X):
        bs = X.size(0)
        X = [X[:,:,10*i:10*i+40] for i in range(13)]#121
        H = torch.zeros(2, bs, 32, device=self.device)
        C = torch.zeros(2, bs, 32, device=self.device)
        for i, x in enumerate(X):
            cnn_out = self.extractor(x) # (bs, ch, seqlen) -> (bs, 32)
            out, (H, C) = self.lstm(cnn_out.unsqueeze(0), (H, C))
        out = H[-1,:,:]
        return self.fc(out)

classifier = CRNN(in_channels=5)

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

model_name = f"SSVAE-SOTA-simple_lr={options.lr}_bs={options.batch_size}_scheduler={options.scheduler}_q-zxy={options.q_zxy}"
if options.sup_samples != 1.0:
	balanced = "B" if options.balanced=="True" else "U"
	model_name += f"_sup-samples={options.sup_samples}-{balanced}"
if options.halpha=="True": 
	model_name += "_halpha"

model_name +="_deriv=sgf"

trainer = SSVAE_Trainer(
	model=model,
	optimizer=optimizer,
	loss_function=Gaussian_NLL(),
	alpha=alpha,
	scheduler=scheduler,
	model_name=model_name,
	save_path="checkpoints/",#"results/temporary/",#
	verbose=True,
	early_stopping=300,
    es_trace=options.es_trace
)

print("everything prepared -> Starting training!")
lh = trainer(range(options.epochs), data_loaders["sup"], data_loaders["unsup"], data_loaders["val"])

trainer.loss_history["encoder_info"] = df.iloc[vae_idx].to_dict() #options.iloc
trainer.loss_history["clf_info"] = "sota"
trainer.loss_history["seed"] = options.seed

# just for prediction
data_loaders = load_and_preprocess_new(mode="sup", transform=H_alpha_only() if options.halpha=="True" else None, batch_size=512, validation=True)

predictor = copy.deepcopy(trainer.model)
predictor.load_state_dict(trainer.early_stopping.best_model)
predictor = predictor.cpu()
predictor.classifier.device = torch.device("cpu")

y_hat, y = acc_tst(predictor.classifier, data_loaders, split="test")
trainer.loss_history["y_pred_test"] = y_hat
trainer.loss_history["y_test"] = y

y_hat, y = acc_tst(predictor.classifier, data_loaders, split="val")
trainer.loss_history["y_pred_val"] = y_hat
trainer.loss_history["y_val"] = y


#torch.save(trainer.loss_history, "/results/temporary/" + trainer.model_name + "_history.pt")
torch.save(trainer.loss_history, "model_histories/" + trainer.model_name + "_history.pt")


