import os
import time
import math
from datetime import datetime

import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from .losses import Gaussian_NLL, gaussian_nll, sample_mse
from .utils import EarlyStopping


class Trainer(nn.Module):
	"""
	Class Trainer was made for easier training of Neural Networks for classification or regression. 
	Insted of defining whole trining precedure every time, it's now possible to do it in 2-3 lines of code.
	"""
	def __init__(self, model, optimizer, loss_function, scheduler=None, **kwargs):
		"""
		Params:
			model: 				Pytorch model
			optimizer: 			Optimizer (e.g. torch.optim.Adam)
			loss_function: 		Type of loss function (nn.CrossEntropyLoss()) 
			scheduler:			Schduler for optimizer's learing rate (Default: None). 
										(e.g. torch.optim.lr_scheduler.MultiStepLR)
			tensorboard:		Boolean parameter (Default: False). Save graph and losses in epochs 
										to tensorboard (True/False).
			model_name:			If tonsorboard==True => model_name is name of tensorboar log.
			set_device:			If you want to use specific device to train NN.(e.g. torch.device("cpu"))
										(if set_device=None => 
											torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
			verbose:			Boolean parameter (Default: False)

		Saving and Stopping methods: (multiple can be used)
			save_every:			If save model.state_dict() every "save_every" epoch.
			early_stopping:		Patience for Early stoping procedure (Default: None). Criterion is validation loss. 
			save_path:			Path to folder where models will be saved. (Defalut: Checkpoints)

		"""
		super(Trainer, self).__init__()
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler

		self.loss_fn = loss_function
		self.loss_history = {"train":[], "validation":[], "val_accuracy":[]} # maybe rather "model_info"

		#optional params
		if kwargs.get("tensorboard") == True:
			self.tensorboard = True
			if kwargs.get("model_name")!= None:
				self.tb = SummaryWriter(comment=kwargs.get("model_name"))
			else:
				self.tb = SummaryWriter()
		else: 
			self.tensorboard = False

		if kwargs.get("early_stopping") != None:
			self.early_stopping = EarlyStopping(kwargs.get("early_stopping"))
			if kwargs.get("save_path") != None:
				self.save_path = kwargs.get("save_path")
			else:
				self.save_path = "checkpoints"
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints")
		else:
			self.early_stopping = None

		if kwargs.get("save_every") != None:
			self.save_every = (kwargs.get("save_every"), True)
			if kwargs.get("save_path") != None:
				self.save_path = kwargs.get("save_path")
			else:
				self.save_path = "checkpoints"
				if not os.path.exists("checkpoints"):
					os.makedirs("checkpoints")
		else:
			self.save_every = (0, False)

		self.model_name = datetime.now().strftime("%d-%m-%Y--%H-%M-%S--") if kwargs.get("model_name")==None \
			else datetime.now().strftime("%d-%m-%Y--%H-%M-%S--") + kwargs.get("model_name")
		self.device = kwargs.get("set_device") if kwargs.get("set_device")!=None \
			else torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)
		self.verbose = True if kwargs.get("verbose")==True else False

		print(self.device) 

	def tensorboard_save(self, epoch, tr_loss, val_loss, acc=None):
		"""
		function for saving informations to tensorboard
		"""
		self.tb.add_scalar("Loss/train", tr_loss, epoch)
		self.tb.add_scalar("Loss/validation", val_loss, epoch)
		"""
		for i in range(len(self.model.model)-1): # iteration through layers
			self.tb.add_histogram(f"layer_{i}/wight", self.model.model[i].layer.weight, epoch)
			self.tb.add_histogram(f"layer_{i}/bias", self.model.model[i].layer.bias, epoch)
			self.tb.add_histogram(f"layer_{i}/wight_grad", self.model.model[i].layer.weight.grad, epoch)
			self.tb.add_histogram(f"layer_{i}/bias_grad", self.model.model[i].layer.bias.grad, epoch)
		"""
		if acc!=None:
			self.tb.add_scalar("Accurary/validation", acc, epoch)

	def tensorboard_graph(self, x):
		if isinstance(self.model, nn.Sequential):
			self.tb.add_graph(self.model, x)
		elif hasattr(self.model, 'model') and isinstance(self.model.model, nn.Sequential):
			self.tb.add_graph(self.model.model, x)
		else:
			self.tb.add_graph(self.model, x)

	def early_stopping_save(self, epoch):
		es_epoch = epoch-(self.early_stopping.global_patience - self.early_stopping.current_patience)
		torch.save(self.early_stopping.best_model, f"{self.save_path}/{self.model_name}_epoch={es_epoch}_early-stop.pt")
		self.loss_history["early_stopping"] = {
			"epoch_idx": es_epoch, 
			"best_loss": self.early_stopping.best_loss,
			"models_saved_as": f"{self.model_name}_epoch={es_epoch}_early-stop.pt"
			}
		return self

	def forward(self, epochs, train_loader, validation_loader):
		n_batches = len(train_loader)
		print_every = max(n_batches//10, 1)

		if not isinstance(epochs, range):
			epochs = range(epochs)
		n_epochs = max(epochs)+1

		for epoch in epochs:
			self.model.train()
			train_loss = 0
			for i, (train_sample, y_true) in enumerate(train_loader, 0):
				train_sample = train_sample.to(self.device)
				y_true = y_true.to(self.device)
				#=================forward================
				y_pred = self.model.forward(train_sample)

				loss = self.loss_fn(y_pred, y_true)
							
				train_loss += loss.detach().item()
				#=================backward===============
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				#=================log====================
				if ((i + 1) % print_every == 0): # and isinstance(history_train_loss, list)
					self.loss_history["train"].append(loss.item())
					if self.tensorboard and epoch==0:
						self.tensorboard_graph(train_sample)

			validation_loss=0
			acc = []
			self.model.eval()
			with torch.no_grad():
				for j, (validation_sample, y_valid_true) in enumerate(validation_loader, 0):
					validation_sample = validation_sample.to(self.device)
					y_valid_true = y_valid_true.to(self.device)
					y_valid_pred = self.model.forward(validation_sample)

					validation_loss += self.loss_fn(y_valid_pred, y_valid_true).detach().item()
					acc.append(accuracy_score(y_valid_true.cpu().detach(), torch.argmax(y_valid_pred.cpu().detach(), axis=1)))

			validation_loss /= len(validation_loader)
			self.loss_history["validation"].append(validation_loss)
			acc = np.mean(acc)
			self.loss_history["val_accuracy"].append(acc)

			if self.verbose:
				print("Epoch [{}/{}], average_loss:{:.4f}, validation_loss:{:.4f}, val_accuracy:{:,.4f}"\
						.format(epoch+1, n_epochs, train_loss/n_batches, validation_loss, acc))
			
			if torch.isnan(torch.tensor(validation_loss)) or torch.isnan(torch.tensor(train_loss)):
				print("Loss is somehow nan!!")
				if self.early_stopping != None:
					self.early_stopping_save(epoch+1)
				return self.loss_history

			if self.tensorboard:
				self.tensorboard_save(epoch=epoch, tr_loss=train_loss/n_batches, val_loss=validation_loss, acc=acc)

			if self.scheduler!=None:
				self.scheduler.step()

			if self.save_every[1]:
				if (epoch+1)%self.save_every[0]==0:
					torch.save(self.model.state_dict(), f"{self.save_path}/{self.model_name}-epoch-{epoch+1}.pt")

			if self.early_stopping != None:
				early_stop = self.early_stopping(self.model.state_dict(), validation_loss)
				if early_stop:
					self.early_stopping_save(epoch+1)
					return self.loss_history
					
		self.early_stopping_save(epoch+1)
		torch.save(self.model.state_dict(), f"{self.save_path}/{self.model_name}_epoch={epoch+1}_end.pt")
		return self.loss_history
	
	def fit_vae(self, epochs, train_loader, validation_loader, samples=1):
		""" later can be merged with forward function """
		n_batches = len(train_loader)
		print_every = max(n_batches//10, 1)

		if not isinstance(epochs, range):
			epochs = range(epochs)
		n_epochs = max(epochs)+1

		for epoch in epochs:
			self.model.train()
			train_loss = 0
			for i, (train_sample, y_true) in enumerate(train_loader, 0):
				train_sample = train_sample.to(self.device)
				y_true = y_true.to(self.device)
				#=================forward================
				loss = - self.model.elbo(x=train_sample, y=y_true, loss_f=self.loss_fn, samples=samples)
				#print(f"epoch-{epoch+1}| iter-{i} | loss {loss.detach().item()}")
							
				train_loss += loss.detach().item()
				#=================backward===============
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				#=================log====================
				if ((i + 1) % print_every == 0): # and isinstance(history_train_loss, list)
					self.loss_history["train"].append(loss.item())
					if self.tensorboard and epoch==0:
						self.tensorboard_graph(train_sample)

			validation_loss=0
			self.model.eval()
			with torch.no_grad():
				for j, (validation_sample, y_valid_true) in enumerate(validation_loader, 0):
					#=========== to device ==============
					validation_sample = validation_sample.to(self.device)
					y_valid_true = y_valid_true.to(self.device)
					#=========== forward pass ===========
					validation_loss -= self.model.elbo(x=validation_sample, y=y_valid_true, loss_f=self.loss_fn, samples=samples).detach().item()

			validation_loss /= len(validation_loader)
			self.loss_history["validation"].append(validation_loss)

			if self.verbose:
				print("Epoch [{}/{}], average_loss:{:.4f}, validation_loss:{:.4f}"\
						.format(epoch+1, n_epochs, train_loss/n_batches, validation_loss))

			if torch.isnan(torch.tensor(validation_loss)) or torch.isnan(torch.tensor(train_loss)):
				print("Loss is somehow nan!!")
				if self.early_stopping != None:
					self.early_stopping_save(epoch+1)
				return self.loss_history

			if self.tensorboard:
				self.tensorboard_save(epoch=epoch, tr_loss=train_loss/n_batches, val_loss=validation_loss)
			if self.scheduler!=None:
				self.scheduler.step()
			if self.save_every[1]:
				if (epoch+1)%self.save_every[0]==0:
					torch.save(self.model.state_dict(), f"{self.save_path}/{self.model_name}-epoch-{epoch+1}.pt")
			if self.early_stopping != None:
				early_stop = self.early_stopping(self.model.state_dict(), validation_loss)
				if early_stop:
					self.early_stopping_save(epoch+1)
					return self.loss_history

		self.early_stopping_save(epoch+1)
		torch.save(self.model.state_dict(), f"{self.save_path}/{self.model_name}-epoch-{epoch+1}-end.pt")
		return self.loss_history


