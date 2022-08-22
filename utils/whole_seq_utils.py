import os
import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.model_selection import train_test_split

from copy import deepcopy
from tqdm import tqdm
from .datasets import UnsupervisedDataset
from .utils import  MultivariateScaler, get_activation, parse_parameters, H_alpha_only
from .models import inception_time, cvae2, rvae_lstm, vae
from .layers import Flatten, ResNetBlock1d, DenseBlock


def load_and_preprocess_whole_seq(path="../", transform=None, batch_size=128, label_idx=80, seed=666, preprocessed=None):
	import os
	import pandas as pd
	assert preprocessed in [None, "sgf"]
	path = os.path.join(path,"data/dataset/")
	path_seq = os.path.join(path, "whole_shot/")
	sd = f"supervised_dataset_{preprocessed}.csv" if preprocessed is not None else "supervised_dataset.csv"
	ud = f"unsupervised_dataset_{preprocessed}.csv" if preprocessed is not None else "unsupervised_dataset.csv"
	sup_dataset = pd.read_csv(os.path.join(path, sd))
	unsup_dataset = pd.read_csv(os.path.join(path, ud))

	X_s, Y, X_u = [], [], []

	for _, seq in sup_dataset.iterrows():
		X_s.append(np.load(os.path.join(path, seq["sequences"])))
		Y.append(np.load(os.path.join(path, seq["labels"])))

	for _, seq in unsup_dataset.iterrows():
		seq_path = seq["sequences"]
		X_u.append(np.load(os.path.join(path, seq_path)))

	X_s = np.vstack(X_s)
	X_u = np.vstack(X_u)
	Y = np.vstack(Y)
	y_s = Y[:,label_idx]

	X_dict, Y_dict = {}, {}

	print(unsup_dataset)
	print(sup_dataset)
	for _, seq in tqdm(unsup_dataset.iterrows()):
		shot_number = seq["sequences"].split("-")[-1][:-4]
		shot_number = shot_number.split("_")[0] if "_" in shot_number else shot_number
		seq_path = seq["sequences"].split(".")[0] + "-whole.npy"
		X_dict[shot_number] = np.load(os.path.join(path_seq, seq_path))

	X_train, X_test, y_train, y_test = train_test_split(np.array(X_s), np.array(y_s), test_size=0.2, random_state=seed)

	#scaling
	scaler = MultivariateScaler(dimension=X_train.shape[1])
	scaler.fit(np.vstack((X_train, X_u)))

	for key in X_dict.keys():
		X_dict[key] = scaler.transform(X_dict[key])

	dataloaders = {}
	print("building dataloaders")
	for key in tqdm(X_dict.keys()):
		dataset = UnsupervisedDataset(X_dict[key], transform=transform)
		dataloaders[key] = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

	return dataloaders


def get_Y_dict(path="../", label_loc=80):
	path = os.path.join(path,"data/dataset/")
	path_seq = os.path.join(path, "whole_shot/")
	#sup_dataset = pd.read_csv(os.path.join(path, "supervised_dataset.csv"))
	unsup_dataset = pd.read_csv(os.path.join(path, "unsupervised_dataset.csv"))

	Y_dict = {}

	for _, seq in tqdm(unsup_dataset.iterrows()):
		try:
			shot_number = seq["sequences"].split("-")[-1][:-4]
			#seq_path = seq["sequences"].split(".")[0] + "-whole.npy"
			lab_path = seq["labels"].split(".")[0] + "-whole.npy"
			#X_dict[shot_number] = np.load(os.path.join(path_seq, seq_path))
			Y_dict[shot_number] = np.load(os.path.join(path_seq, lab_path))
			Y_dict[shot_number] = Y_dict[shot_number][:,label_loc]
		except:
			print(seq, "failed")
	return Y_dict


class Rebuilder(object):
	def __init__(self, model_name, cuda=True):
		self.ss = "SSVAE" in model_name
		self.model_name = model_name
		self.parameters = model_name.split("--")[-1]
		self.model_type = self.parameters.split("_")[0]	
		self.cuda = cuda
		self.model = None

	def InceptionTime(self):
		par = parse_parameters(self.parameters)

		if "activation" not in par.keys():
			par["activation"] = "swish"

		model = inception_time(
			xdim=5, #5 halpha only
			ydim=4, 
			n_filters=par["nfilters"], 
			fsizes=par["fsize"], 
			bottleneck=par["bneck"], 
			blocks=par["blocks"], 
			activation=par["activation"]
		)
		self.model = model
		#return self
		return model

	def CRNN(self):
		class CRNN(nn.Module):
			def __init__(self, device=torch.device("cuda")):
				super(CRNN, self).__init__()
				self.extractor = nn.Sequential(
					nn.Conv1d(5,64,3,padding=1), # 5
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

		model = CRNN(torch.device("cuda" if self.cuda else "cpu"))
		self.model = model
		#return self
		return  model

	def ResNet(self):#TODO 
		par = parse_parameters(self.parameters)
		depth = len(par["nfilters"])
		if depth == 5:
			#nn.Sequential(
			model = nn.Sequential(
				nn.Conv1d(in_channels=5, out_channels=par["nfilters"][0], kernel_size=7, padding=3),
				nn.BatchNorm1d(par["nfilters"][0]),
				get_activation(par["activation"])(),
				ResNetBlock1d(in_channels=par["nfilters"][0], out_channels=par["nfilters"][1], kernel_size=3, layers=2, activation=get_activation(par["activation"])),
				ResNetBlock1d(in_channels=par["nfilters"][1], out_channels=par["nfilters"][2], kernel_size=3, layers=2, activation=get_activation(par["activation"])),
				ResNetBlock1d(in_channels=par["nfilters"][2], out_channels=par["nfilters"][3], kernel_size=3, layers=2, activation=get_activation(par["activation"])),
				#nn.MaxPool1d(2),
				ResNetBlock1d(in_channels=par["nfilters"][3], out_channels=par["nfilters"][4], kernel_size=3, layers=2, activation=get_activation(par["activation"])),
				nn.AdaptiveAvgPool1d(output_size=1),
				Flatten(out_features=par["nfilters"][-1]),
				nn.Linear(in_features=par["nfilters"][-1], out_features=4)
			)
		elif depth==3:
			model = nn.Sequential(
				nn.Conv1d(in_channels=5, out_channels=par["nfilters"][0], kernel_size=7, padding=3),
				nn.BatchNorm1d(par["nfilters"][0]),
				get_activation(par["activation"])(),
				ResNetBlock1d(in_channels=par["nfilters"][0], out_channels=par["nfilters"][1], kernel_size=3, layers=2, activation=get_activation(par["activation"])),
				nn.MaxPool1d(2),
				ResNetBlock1d(in_channels=par["nfilters"][1], out_channels=par["nfilters"][2], kernel_size=3, layers=2, activation=get_activation(par["activation"])),
				nn.AdaptiveAvgPool1d(output_size=1),
				Flatten(out_features=par["nfilters"][-1]),
				nn.Linear(in_features=par["nfilters"][-1], out_features=4)
			)
		else:
			raise ValueError("unknown ResNet config")
		self.model = model
		return model

	def second_stage(self, model, par):
		in_, out_ = model[-1].mu.in_features, model[-1].mu.out_features 

		if par["hidden"] == 0:
			model[-1] = nn.Linear(in_features=in_, out_features=4) # out_ == zdim
		else:
			model[-1] = nn.Linear(in_features=in_, out_features=out_) # out_ == zdim
			model = [model]
			hiddens = par["hidden"] if isinstance(par["hidden"], list) else [par["hidden"]] 
			hidden_neurons = [out_]+hiddens
			for (i, o) in zip(hidden_neurons[:-1], hiddens):
				model.append(
					DenseBlock(
						input_dim=i,
						output_dim=o,
						activation=get_activation(par["activation"])(),
						batch_norm=bool(par["bnorm"]),
						dropout=par["dropout"]
					)
				)
			model.append(
				nn.Linear(in_features=hiddens[-1], out_features=4)
			)
		model = nn.Sequential(*model)
		return model

	def CNN(self, config=0, zdim=15):	
		par = parse_parameters(self.parameters)
		if config==4:
			model = cvae2(
					xdim=[5,160], 
					zdim=zdim, 
					n_filters=[128,128,128,128], 
					fsizes=[5,3,3,3], 
					stride=[1,1,1,1], 
					padding=[2,1,1,1], 
					pooling=[0,2,0,2], 
					dense= None, 
					activation=get_activation("swish"), 
					ydim=0, 
					q_zxy=True, 
					upsample="conv"
				)
		elif config==3:
			model = cvae2(
					xdim=[5,160], 
					zdim=zdim, 
					n_filters=[64,128,128,128], 
					fsizes=[5,3,3,3], 
					stride=[1,1,1,1], 
					padding=[2,1,1,1], 
					pooling=[0,2,0,2], 
					dense= 256, 
					activation=get_activation("swish"), 
					ydim=0, 
					q_zxy=True, 
					upsample="conv"
				)
		elif config==5:
			model = cvae2(
					xdim=[5,160], 
					zdim=zdim, 
					n_filters=[64,128,128,128], 
					fsizes=[5,3,3,3], 
					stride=[1,1,1,1], 
					padding=[2,1,1,1], 
					pooling=[0,2,0,2], 
					dense= None, 
					activation=get_activation("swish"), 
					ydim=0, 
					q_zxy=True, 
					upsample="conv"
				)
		elif config==0:
			model = cvae2(
					xdim=[5,160], 
					zdim=zdim, 
					n_filters=[128,128,128], 
					fsizes=[6,4,4], 
					stride=[2,2,2], 
					padding=[0,0,0], 
					pooling=[0,0,0], 
					dense= 256, 
					activation=get_activation("swish"), 
					ydim=0, 
					q_zxy=True, 
					upsample="conv"
				)
		else:
			raise ValueError("wrong config")
		model = deepcopy(model.encoder)
		model = self.second_stage(model, par)
		self.model = model
		return model

	def RNN(self):
		par = parse_parameters(self.parameters)

		model = rvae_lstm(
			xdim=[5,160],
			zdim=30, 
			hidden_size=256,
			activation=get_activation("leaky")
		) 
		
		model = deepcopy(model.encoder)
		model = self.second_stage(model, par)
		self.model = model
		return model

	def FNN(self):
		par = parse_parameters(self.parameters)
		if (par["encoder-idx"] == 12) or (par["encoder-idx"] == 10):
			model = vae(
				xdim = [5,160],
				zdim = 15, 
				hidden_neurons=[128, 128, 128], 
				batch_norm=False, 
				activation=get_activation("swish"), 
				dropout= 0,
				ydim=0,
				q_zxy=False
			)
		else: 
			raise ValueError("unknown vae index")
		model = deepcopy(model.encoder)
		model = self.second_stage(model, par)
		self.model = model
		return model

	def load_state_dict(self, model, verbose=False):
		sd = torch.load("checkpoints/" + self.model_name, map_location=torch.device("cpu"))
		if self.ss:
			if verbose:
				print("SS loading")
			clf_keys=list(filter(lambda x: "classifier" in x, sd.keys())) # len(classifier.)=11 -> [11:]
			clf_sd = { your_key[11:]: sd[your_key] for your_key in clf_keys}
			model.load_state_dict(clf_sd)
		else:
			if verbose:
				print("Standard clf loading")
			model.load_state_dict(sd)
		return model

	def sum_parameters(self, model):
		s = []
		for p in model.parameters():
			dims = p.size()
			n = np.prod(p.size())
			s.append((dims, n))
		return s, np.sum([j for i,j in s])

	def __call__(self):
		if "InceptionTime" in self.model_type:
			model = self.InceptionTime()
			model = self.load_state_dict(model)
		elif "SOTA" in self.model_type:
			model = self.CRNN()
			model = self.load_state_dict(model)
		elif "ResNet" in self.model_type:
			model = self.ResNet()
			model = self.load_state_dict(model)
		elif "SSVAE-CNN+VAE-simple" == self.model_type:
			try:
				model = self.CNN(config=0)
				model = self.load_state_dict(model)
			except RuntimeError:
				try: 
					model = self.CNN(config=4)
					model = self.load_state_dict(model)
				except RuntimeError:
					try: 
						model = self.CNN(config=5)
						model = self.load_state_dict(model)
					except RuntimeError:
						try: 
							model = self.CNN(config=3)
							model = self.load_state_dict(model)
						except:
							raise IndexError("wrong config for CNN classifier")
			
		elif "SSVAE-simple-hybrid" in self.model_type:
			model = self.RNN()
			model = self.load_state_dict(model)
		elif "SSVAE-simple" == self.model_type:
			par = parse_parameters(self.parameters)
			if par["encoder-idx"]==3:
				model = self.CNN(config=3, zdim=15)
				model = self.load_state_dict(model)
			elif par["encoder-idx"]==0:
				model = self.CNN(config=0, zdim=15)
				model = self.load_state_dict(model)
			else:
				model = self.FNN()
				model = self.load_state_dict(model)
		elif "LDM-based" == self.model_type:
			par = parse_parameters(self.parameters)
			if (par["encoder-idx"]==10) or (par["encoder-idx"]==12):
				model = self.FNN()
				model = self.load_state_dict(model)
			elif (par["encoder-idx"]==5) or (par["encoder-idx"]==6):
				model = self.RNN()
				model = self.load_state_dict(model)
			elif par["encoder-idx"]==0:
				model = self.CNN(config=0, zdim=15)
				model = self.load_state_dict(model)
			elif par["encoder-idx"]==4:
				model = self.CNN(config=5, zdim=15)
				model = self.load_state_dict(model)
			elif par["encoder-idx"]==3:
				model = self.CNN(config=3, zdim=15)
				model = self.load_state_dict(model)
			else:
				print("Unkown LDM-based config")
		else:
			print("Unknown model")
		return self