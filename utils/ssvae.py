import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

from copy import copy, deepcopy
from .layers import *
from .utils import One_Hot, dict_append
from .losses import Gaussian_NLL
from .inference import Trainer


# Semi-supervised Variational Autoencoder

class SSVAE(nn.Module):
	def __init__(self, encoder, decoder, classifier):
		super(SSVAE, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.classifier = classifier

	def forward(self, x, y=None):
		if y == None:
			y = F.softmax(self.classifier(x), dim=1)
			y = torch.eye(y.size(1), device=y.device)[torch.argmax(y, dim=1)] # one hot encoding
			#y = One_Hot(y.size(1), y.device)(torch.argmax(y, dim=1))

		z, mu, sigma = self.encoder(x, y) # q(z|x,y) or q(z|x)
		zy = torch.cat((z,y), axis=1)
		return self.decoder(zy), mu, sigma

	def elbo_xy(self, x, y, loss_f=Gaussian_NLL()):
		z, mu, sigma = self.encoder(x, y)
		#sigma = simga + 1e-20 # add stability ?
		kld_z = - 0.5 * torch.mean(torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2), axis=1)) # +1e-20 for stability

		zy = torch.cat((z,y), axis=1)
		likelihood  = - loss_f(self.decoder(zy), x)
		return likelihood - kld_z + math.log(1/y.size(1))

	def elbo_x(self, x, loss_f=Gaussian_NLL()): 
		y = F.softmax(self.classifier(x), dim=1)
		L_xy = []
		ydim = y.size(1)
		y_oh = One_Hot(ydim, y.device)
		for i in range(ydim):
			#y_oh = torch.eye(ydim, device=y.device)[i] #One_Hot(ydim, y.device)(i)
			L_xy.append(self.elbo_xy(x, y_oh(i, y.size(0)), loss_f=loss_f).reshape(1,1)) #y_oh
		L_xy = torch.cat(L_xy)
		#print(f"L_xy.shape = {L_xy.shape} || y@L_xy .shape = {(y@L_xy).shape} || H(q) = {torch.mean(y*torch.log(y), dim=1).shape}")
		Lx = torch.squeeze(y@L_xy) - torch.mean(y*torch.log(y+1e-20), dim=1) # to avoid nans
		return torch.mean(Lx)

	def loss(self, x_s, y_s, x_u, alpha=1, loss_f=Gaussian_NLL(), clf_loss=nn.CrossEntropyLoss()): 
		l_xy = self.elbo_xy(x_s, y_s, loss_f=loss_f)
		clf_loss = clf_loss(self.classifier(x_s), y_s.argmax(dim=1)) # neg log likelihood 
		l_x = self.elbo_x(x_u, loss_f=loss_f)
		loss = - l_xy - l_x + alpha * clf_loss 
		return loss, (l_xy, l_x, clf_loss) # for logging of losses 


class SSVAE_V2(SSVAE):
	"""
	Semi-supervised VAE with shared feature extraction 
	"""
	def __init__(self, core, enc_z, enc_y, dec):
		encoder = stacked_encoder(
			core,
			enc_z
		)
		classifier = stacked_classifier(
			core,
			enc_y
		)
		super(SSVAE_V2, self).__init__(encoder, dec, classifier)


# wrappers for encoders q(z|x) and q(z|x,y) etc.

class stacked_encoder(nn.Module):
	def __init__(self, core, enc_z):
		super(stacked_encoder, self).__init__()
		self.core = core
		self.enc_z = enc_z
	
	def forward(self, x, y=None):
		h = self.core(x)
		return self.enc_z(h)


class stacked_classifier(nn.Module):
	def __init__(self, core, enc_y):
		super(stacked_classifier, self).__init__()
		self.core = core
		self.enc_y = enc_y
	
	def forward(self, x):
		h = self.core(x)
		return self.enc_y(h)


class q_zx(nn.Module):
	def __init__(self, encoder):
		super(q_zx, self).__init__()
		self.encoder = encoder

	def forward(self, x, y=None):
		return self.encoder(x)


class q_zxy(nn.Module):
	def __init__(self, dense, conv=None):
		"""
		Encoder for semi-supervised VAE -> q(z|x,y)

		Params:
			dense:		Part of encoder which consist of dense layers
			conv:		Convolutional feature extractor (optional). 
						(x -> conv -> dense -> z)

		Example:
			>>> from utils.layers import VariationalLayer, Flatten
			>>> from utils.ssvae import q_zxy

			>>> x = torch.randn(5, 3, 10)
			>>> y = torch.randint(4,(5,))
			>>> y
			tensor([1, 3, 3, 2, 1])
			>>> y_one_hot = torch.eye(4)[y]
			>>> y_one_hot
			tensor([[0., 1., 0., 0.],
				[0., 0., 0., 1.],
				[0., 0., 0., 1.],
				[0., 0., 1., 0.],
				[0., 1., 0., 0.]])
			>>> encoder = q_zxy(dense = VariationalLayer(in_features=3*10+4, out_features=3) , conv=Flatten(out_features=3*10))
			>>> z, mu, sigma = encoder(x, y_one_hot)
			>>> (z.shape, mu.shape, sigma.shape)
			(torch.Size([5, 3]), torch.Size([5, 3]), torch.Size([5, 3]))
		"""
		super(q_zxy, self).__init__()
		self.dense = dense
		self.conv = conv if conv != None else pass_through # pass_throught(x) = x

	def forward(self, x, y):
		x = self.conv(x) # already should be flattened
		xy = torch.cat((x,y), dim=-1)
		return self.dense(xy)


class SSVAE_Trainer(Trainer):
	def __init__(self, model, optimizer, loss_function=Gaussian_NLL(), alpha=1, scheduler=None, model_name=None, verbose=False, early_stopping=None, es_trace="elbo_loss", save_path=None):
		"""
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		Params:
			model: 				SSVAE model
			optimizer: 			Optimizer (e.g. torch.optim.Adam)
			loss_function: 		Type of loss function Gaussian_NLL() or torch.nn.MSELoss()
			scheduler:			Schduler for optimizer's learing rate (Default: None). 
										(e.g. torch.optim.lr_scheduler.MultiStepLR)
			model_name:			If tonsorboard==True => model_name is name of tensorboar log.
			verbose:			Boolean parameter (Default: False)

		Saving and Stopping methods: (multiple can be used)
			early_stopping:		Patience for Early stoping procedure (Default: None). Criterion is validation loss. 
			save_path:			Path to folder where models will be saved. (Defalut: Checkpoints)

		"""
		super(SSVAE_Trainer, self).__init__(
			model=model, 
			optimizer=optimizer, 
			loss_function=loss_function, 
			scheduler=scheduler, 
			tensorboard=False, 
			model_name=model_name, 
			verbose=verbose, 
			early_stopping=early_stopping, 
			save_path=save_path,
			#set_device=torch.device("cpu") # TODO remove
		)
		self.loss_history = {
			"train":{"epoch": [], "iter": [], "loss":[], "L_xy":[], "L_x":[], "clf":[]}, 
			"valid":{"epoch": [], "iter": [], "loss":[], "L_xy":[], "L_x":[], "clf":[], "acc":[]}, 
			"alpha":alpha
		}
		self.alpha = alpha
		assert es_trace in ["elbo_loss", "clf_loss"]
		self.es_trace = es_trace

	def early_stopping_save(self, iters):
		"""
		Epochs are used for training, but early stopping is evaluated 10 times per one epochs => for saving are better \"iterations\"
		"""
		es_iter = iters-(self.early_stopping.global_patience - self.early_stopping.current_patience)
		torch.save(self.early_stopping.best_model, f"{self.save_path}/{self.model_name}_iters={es_iter}_early-stop.pt")
		self.loss_history["early_stopping"] = {
			"iter_idx": es_iter, 
			"best_loss": self.early_stopping.best_loss,
			"es_trace": self.es_trace,
			"models_saved_as": f"{self.model_name}_iters={es_iter}_early-stop.pt"
			}
		return self

	def forward(self, epochs, sup_loader, unsup_loader, val_loader):
		"""
		Epochs are used for training, but early stopping is evaluated 10 times per one epoch.
		Reason for this is great difference between number of labeled vs unlabeled samples. 
		While creating data loaders it is better to randomly sample from supervised dataset (without replacement).
		=> during training every sample from unsupervised dataset is chosen only once but labeled sample can be selected muliple times!
		=> only iterations would be more suitable for this training (to randomly sample from both datasets), 
			but previous models were trained epoch-wise. So this is just to be little consistent.
		"""

		n_batches = len(unsup_loader)
		self.loss_history["iters_per_epoch"] = n_batches
		print_every = max(n_batches//100, 1)
		valid_every = max(n_batches//10, 1)

		if not isinstance(epochs, range):
			epochs = range(epochs)
		n_epochs = max(epochs)+1

		for epoch in epochs:
			for it, ((x_s, y_s), x_u) in enumerate(zip(sup_loader, unsup_loader)):
				self.model.train()
				x_s, y_s, x_u = x_s.to(self.device), y_s.to(self.device), x_u.to(self.device)
				#=================forward================
				loss, log = self.model.loss(x_s, y_s, x_u, self.alpha, loss_f=self.loss_fn) # clf_loss=nn.CrossEntropyLoss()
				#=================backward===============
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				#set_trace()
				#=================log====================
				if (it % print_every == 0):
					dict_append(
						self.loss_history["train"], 
						["epoch", "iter", "loss", "L_xy", "L_x", "clf"],
						[epoch, it, loss.item(), log[0].item(), log[1].item(),log[2].item()]
					)
					if self.verbose:
						print(f"train: iter = {epoch*n_batches+it} | Loss = {loss.item():.2f} | L_xy = {log[0]:.2f} | L_x = {log[1]:.2f} | clf = {log[2]:.2f} ")

				if (it % valid_every == 0):
					val_loss={}
					self.model.eval()
					with torch.no_grad():
						for j, (x_vs, y_vs) in enumerate(val_loader, 0):
							x_vs, y_vs = x_vs.to(self.device), y_vs.to(self.device)
							loss, log = self.model.loss(x_vs, y_vs, x_vs, self.alpha, loss_f=self.loss_fn)
							y_pred = self.model.classifier(x_vs)
							#set_trace()
							acc = torch.mean(torch.tensor(y_pred.cpu().detach().argmax(dim=1)==y_vs.cpu().detach().argmax(dim=1), dtype=float)).item()
							dict_append(
								val_loss, 
								["loss", "L_xy", "L_x", "clf", "acc"],
								[loss.item(), log[0].item(), log[1].item(), log[2].item(), acc]
							)
					loss_, lxy_, lx_, clf_, acc_ = np.mean(val_loss["loss"]), np.mean(val_loss["L_xy"]), np.mean(val_loss["L_x"]), np.mean(val_loss["clf"]), np.mean(val_loss["acc"])
					dict_append(
						self.loss_history["valid"], 
						["epoch", "iter", "loss", "L_xy", "L_x", "clf", "acc"],
						[epoch, it, loss_, lxy_, lx_, clf_, acc_]
					)
					if self.verbose:
						print(f"validation: iter = {epoch*n_batches+it} | Loss = {loss_:.2f} | L_xy = {lxy_:.2f} | L_x = {lx_:.2f} | clf = {clf_:.2f} | acc = {acc_} ")

					if self.early_stopping != None:
						es_loss = loss_ if self.es_trace=="elbo_loss" else clf_
						early_stop = self.early_stopping(self.model.state_dict(), es_loss) # loss_
						if early_stop:
							self.early_stopping_save(epoch*n_batches+it)
							return self.loss_history

				last_val_loss = self.loss_history["valid"]["loss"][-1]
				last_tr_loss = self.loss_history["train"]["loss"][-1]

				if torch.isnan(torch.tensor(last_val_loss)) or torch.isnan(torch.tensor(last_tr_loss)):
					print(f"Loss is somehow nan!! epoch={epoch}| iter={it}")
					if self.early_stopping != None:
						self.early_stopping_save(epoch*n_batches+it)
					return self.loss_history

			if self.scheduler!=None:
				self.scheduler.step()
		
		self.early_stopping_save(epoch*n_batches+it) # To have saved best model as well as last model
		torch.save(self.model.state_dict(), f"{self.save_path}/{self.model_name}_iter={(epoch+1)*n_batches}_end.pt")
		return self.loss_history
