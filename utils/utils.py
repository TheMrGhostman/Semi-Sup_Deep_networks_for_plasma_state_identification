import numpy as np 
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from copy import deepcopy, copy


class One_Hot(object):
	def __init__(self, n_classes, device="cpu"):
		self.n_classes = n_classes
		self.class_matrix = torch.eye(n_classes, device=device)

	def __call__(self, p, rows=None):
		if rows == None:
			return self.class_matrix[p]
		else:
			return torch.ones(rows, self.n_classes, device=self.class_matrix.device) * self.class_matrix[p]


class SWISH(nn.Module):
	def __init__(self):
		super(SWISH, self).__init__()

	def forward(self, x):
		return x * torch.sigmoid(x)


class SafeSoftplus(nn.Module):
	def __init__(self, stability_const=1e-20):
		super(SafeSoftplus, self).__init__()
		self.softplus = nn.Softplus()
		self.stability_const = stability_const # output will be always greater then zero
	
	def forward(self, x):
		return self.softplus(x)+self.stability_const


class EarlyStopping(object):
	def __init__(self, patience):
		self.global_patience = patience
		self.current_patience = patience
		self.best_loss = np.Inf
		self.best_model = None # model or model_state_dict
	
	def __call__(self, model, new_loss):
		if new_loss < self.best_loss:
			self.best_loss = new_loss
			self.best_model = deepcopy(model)
			self.current_patience = copy(self.global_patience)
		else:
			self.current_patience -= 1

		if self.current_patience == 0:
			return True # stop training
		else:
			return False # continue training
	
	def reset(self):
		self.current_patience = copy(self.global_patience)


class MultivariateScaler(object):
	def __init__(self, scaler=RobustScaler, dimension=5):
		self.scaler = scaler
		self.scalers = {} # each dimension has its own scaler
		self.dimension = dimension

	def fit(self, X):
		assert self.dimension==X.shape[1] # check number of channels
		for i in range(self.dimension):
			self.scalers[i] = self.scaler()
			self.scalers[i].fit(X[:, i, :]) 
		return self

	def transform(self, X):
		assert self.dimension==X.shape[1] # check number of channels
		for i in range(self.dimension):
			X[:, i, :] = self.scalers[i].transform(X[:, i, :]) 
		return X

	def fit_transform(self, X):
		assert self.dimension==X.shape[1] # check number of channels
		for i in range(self.dimension):
			self.scalers[i] = self.scaler()
			self.scalers[i].fit(X[:, i, :]) 
			X[:, i, :] = self.scalers[i].transform(X[:, i, :]) 
		return self


class H_alpha_only(object):
	def __init__(self, channel=0):
		"""
		old dataset -> H_alpha channel is 2
		new dataset -> H_alpha channel is 0
		"""
		self.channel = channel

	def __call__(self, x):
		"""
		Input:
			x 		... 	(BS, channels, sequence_len)
		Output:
			x_hat 	...		(BS, 1, sequence_len)
		"""
		x_hat = x[self.channel, :] # (BS, sequence_len)
		x_hat = x_hat.unsqueeze(0) # (BS, 1, sequence_len)
		return x_hat


def correct_sizes(sizes):
	corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
	return corrected_sizes


def compute_dims_1d(L_in, ksizes, strides=1, padding=0, pooling=1, check_for_error=False):
	"""
	: param L_in			Dimensions of input (multivariate sequences).
	: param ksizes			List of kernel sizes
	: param strides 		List (or int) of strides.
	: param padding			List (or int) of paddings

	Example:
		>>> compute_dims_1d(L_in=160, ksizes=[3,3,3], strides=[2,2,2], padding=0)
		([160, 79, 39, 19], [160, 79.5, 39.0, 19.0])
		
	"""
	def conv_output(L_in, ksize, stride=1, padding=0):
		L_out = (L_in + 2*padding - ksize)/stride + 1
		return L_out, int(L_out)

	strides = strides*np.ones(len(ksizes)) if isinstance(strides, int) else strides
	padding = padding*np.ones(len(ksizes)) if isinstance(padding, int) else padding
	pooling = pooling*np.ones(len(ksizes)) if isinstance(pooling, int) else pooling
	
	dims = [L_in]
	dims_float = [L_in]
	for (k,s,p,pool) in zip(ksizes, strides, padding, pooling):
		d_f, d = conv_output(dims[-1], k, s, p)
		if pool!=0:
			d_f = d_f/pool
			d = int(d/pool)
		dims.append(d)
		dims_float.append(d_f)
	

	if check_for_error and ([float(x) for x in dims] != dims_float):
		print(dims, dims_float)
		raise ValueError("Input lenght was rounded. Parameters unfit for decoder!")
	
	return dims, dims_float


def paremeters_summary(model):
	s = []
	for p in model.parameters():
		dims = p.size()
		n = np.prod(p.size())
		s.append((dims, n))
	return s, np.sum([j for i,j in s])


def plot_loss(obj, figsize=(25,18), downsample=None):
	"""
	: param obj: 	Object type SVI or Trainer
	"""
	loss_train = obj.loss_history["train"]
	axe_t = np.arange(len(loss_train))/10
	loss_val = obj.loss_history["validation"]
	axe_v = np.arange(len(loss_val))
	if downsample!=None:
		axe_t = axe_t[::downsample]
		loss_train = loss_train[::downsample]
	plt.figure(figsize=figsize)
	plt.plot(axe_t, loss_train, lw=0.5)
	plt.plot(axe_v, loss_val, lw=0.5)
	plt.ylabel("loss")
	plt.xlabel("Epochs")


	if "val_accuracy" in obj.loss_history.keys():
		print("plotting accuracy")
		plt.figure("Accuracy", figsize=figsize)
		plt.plot(obj.loss_history["val_accuracy"])
		plt.ylabel("Accuracy")
		plt.ylim(0, 1)
		plt.grid(True)
	plt.show()


def get_activation(string):
	if string == "relu":
		activation = nn.ReLU
	elif string == "prelu":
		activation = nn.PReLU
	elif string == "leaky":
		activation = nn.LeakyReLU
	elif string == "swish":
		activation = SWISH
	else:
		raise ValueError("unknown activation function")
	return activation


def dict_append(dict_, keys, values):
	for (k,v) in zip(keys, values):
		if k not in dict_.keys():
			dict_[k] = []
		dict_[k].append(v)
	return dict_


def acc_tst(model, data_loaders, split="test"):
	y_pred=[]
	y_=[]
	model.eval()
	with torch.no_grad():
		for (x,y) in data_loaders[split]:
			y_.append(y)
			y_p = model(x)
			y_pred.append(y_p.argmax(axis=1).numpy())
		#print(np.mean(np.hstack(y_pred)==np.hstack(y_)))
	return np.hstack(y_pred), np.hstack(y_)


def parse_parameters(parameters):
	parameters = parameters.replace("\t","")
	parameters = parameters.replace(" ", "")
	parsed_params = {}
	for p in parameters.split("_"):
		if "=" in p:
			k,v = p.split("=")
			try:
				v = int(v)
			except:
				try:
					v = float(v)
				except:
					try:
						if len(v.split("-"))!=1:
							v = [int(e) for e in v.split("-")]
						else:
							v = str(v)
					except:
						try:
							if len(v.split("-"))!=1:
								v = [float(e) for e in v.split("-")]
							else:
								v = str(v)
						except:
							try:
								if len(v.split("-"))==2:
									v = v.split("-")
									v = [float(v[0]), str(v[1])]
								else:
									v = str(v)
							except:
								if (k=="channels") and (v==["non", "phys"]):
									v = "non-phys"
								else:
									raise ValueError(f"something went wrong?! -> {k}={v}" )

			parsed_params[k] = v
	return parsed_params
