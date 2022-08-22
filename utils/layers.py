import numpy as np 
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F 

from collections import OrderedDict
from IPython.core.debugger import set_trace
from torch.nn.modules.batchnorm import BatchNorm1d

# Helper Layers

class Flatten(nn.Module):
	def __init__(self, out_features):
		super(Flatten, self).__init__()
		self.output_dim = out_features

	def forward(self, x):
		return x.view(-1, self.output_dim)


class Reshape(nn.Module):
	def __init__(self, out_shape):
		super(Reshape, self).__init__()
		self.out_shape = out_shape

	def forward(self, x):
		return x.view(-1, *self.out_shape)


def pass_through(X):
	return X


# Layers

class VariationalLayer(nn.Module):
	def __init__(self, in_features, out_features, bias=True, return_KL=False):
		"""
		Variational Layer with reparametrization trick. 
		It's used as bottleneck of Variational AutoEncoder ie. output of encoder.
		\"Linear layer for variational inference with reparametrization trick\"

		Reparametrization trick and computations of KL_divergence is included for compatibility reasons.

		Parameters:
			in_features: 		Number of input features (number of neurons on input)
			out_features:		Number of output features (number of neurons on output)
			bias:				Include bias - True/False
			return_KL			Compute and return KL divergence - True/False (old models need it)

		
		"""
		super(VariationalLayer, self).__init__()
		self.mu = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
		self.rho = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
		self.softplus = nn.Softplus()
		self.return_KL = return_KL

	def forward(self, x_in):
		mu = self.mu(x_in)
		sigma = self.softplus(self.rho(x_in)) + 1e-10 # improves stability 
		eps = torch.randn_like(sigma)
		if self.return_KL:
			KL_div = - 0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) # kl_div 
			return mu + sigma*eps, KL_div, mu, sigma
		return mu + sigma*eps, mu, sigma


class VariationalDecoderOutput(nn.Module):
	"""
	Variational Layer where variances are same and not equal to 1
	
	2D example:
		mu = (mu_1, mu_2, mu_3).T

		C = (sigma, 0    , 0    )
			(0    , sigma, 0    )
			(0    , 0    , sigma)
	"""

	def __init__(self, in_features, mu_out, sigma_out=1, bias=True, reshape=False):

		"""	
		Params:
			in_features 		Number of input features (number of neurons on input)
			mu_out				Number of mu's output features (number of neurons on output)
			sigma_out			Dimension of sigma output
			bias				Include bias - True/False
		"""
		super(VariationalDecoderOutput, self).__init__()
		self.mu = nn.Linear(in_features=in_features, out_features=mu_out, bias=bias)
		self.rho = nn.Linear(in_features=in_features, out_features=sigma_out, bias=bias)
		self.softplus = nn.Softplus()
		self.reshape_mu = Reshape(out_shape=(sigma_out, mu_out//sigma_out)) if reshape else pass_through
		self.reshape_sigma = Reshape(out_shape=(sigma_out, 1)) if reshape else pass_through

	def forward(self, x):
		mu = self.mu(x)
		sigma = self.softplus(self.rho(x)) + 1e-10 # improves stability 
		return self.reshape_mu(mu), self.reshape_sigma(sigma)


class ConvDecoderOutput(nn.Module):
	def __init__(self, in_channels, in_features, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
		super(ConvDecoderOutput, self).__init__()
		self.mu = nn.ConvTranspose1d(
				in_channels=in_channels, 
				out_channels=out_channels, 
				kernel_size=kernel_size, 
				stride=stride, 
				padding=padding,
				bias=bias
			)

		# Conv1d with kernels size same as input equals to dense layer
		# but we dont have to take care of reshaping with this approach
		self.rho = nn.Conv1d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=in_features,
			stride=1,
			bias=bias
			)
		
		self.softplus = nn.Softplus()

	def forward(self, x):
		"""
		x must be shape ... (-1, in_channels, size)
		"""
		mu = self.mu(x)
		sigma = self.softplus(self.rho(x)) + 1e-10 # improves stability 
		return mu, sigma


class ConvTransposeDecoderOutput(nn.Module):
	def __init__(self, in_channels, in_features, out_features, kernel_size, stride=1, padding=0, bias=True):
		super(ConvTransposeDecoderOutput, self).__init__()
		self.mu = nn.ConvTranspose1d(
				in_channels=in_channels, 
				out_channels=1, 
				kernel_size=kernel_size, 
				stride=stride, 
				padding=padding
			)
		self.rho = nn.Linear(in_features=in_features, out_features=1, bias=bias)
		self.flatten = Flatten(out_features=in_features)
		self.flatten_mu = Flatten(out_features=out_features)
		self.softplus = nn.Softplus()

	def forward(self, x_in):
		"""
		x must be shape ... (-1, in_channels, size)
		"""
		mu = self.mu(x_in)
		x = self.flatten(x_in)
		sigma = self.softplus(self.rho(x)) + 1e-10 # improves stability 
		return self.flatten_mu(mu), sigma


class RecurrentDecoderOutput(nn.Module):
	def __init__(self, in_features, sequence_len, out_features, bias=True):
		"""	
		Params:
			param in_features: 		Number of input features (number of neurons on input)
			param sequence_len		Length of sequence
			param out_features 		Number of output features
			param bias:				Include bias - True/False
		"""
		super(RecurrentDecoderOutput, self).__init__()

		self.in_features = in_features
		self.sequence_len = sequence_len
		self.mu = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
		self.rho = nn.Linear(in_features=in_features*sequence_len, out_features=out_features, bias=bias)
		self.softplus = nn.Softplus()
		self.reshape_sigma = Reshape(out_shape=(out_features, 1))

	def forward(self, x_in):
		"""
		x must be shape ... (sequence_len, batch_size, in_features) 
		"""
		# TODO need to be tested
		mu = self.mu(x_in) # now is shape (sequence_len, bactch_size, n_features)
		mu = mu.permute(1,2,0)

		x_in = x_in.permute(1,2,0).reshape(-1, self.sequence_len*self.in_features) #(batch_size, sequence_len*in_features) = (100, 128*160)
		sigma = self.softplus(self.rho(x_in))
		sigma = self.reshape_sigma(sigma)
		return mu, sigma


class DenseBlock(nn.Module):
	"""
	Simple linear block layer consisting of linear layer and activation function
	"""
	def __init__(self, input_dim, output_dim, activation=nn.ReLU(), batch_norm=False, dropout=False, bias=True):
		"""
		: param input_dim		Number of input features (number of neurons from previos layer).
		: param output_dim		Number of output featuers (number of neurons of output)
		: param activation 		Activation function for this layer (nn.ReLU() <– object type)
		: param batch_norm		Boolean to use batch normalization or not (True/False)
		: param dropout         Probability of dropout p. (dropout==False -> no dropout)
		"""
		super(DenseBlock, self).__init__()
		self.layer = nn.Linear(in_features=input_dim, out_features=output_dim, bias=bias)
		self.batch_norm =  nn.BatchNorm1d(num_features=output_dim) if batch_norm else pass_through
		self.activation = activation
		self.dropout = nn.Dropout(p=dropout) if dropout!=False else pass_through

	def forward(self, x):
		"""
		: param x 			Input data (N, L) = (N samples in batch, Length of input featues per sample). (batch of data)
		"""
		x = self.layer(x)
		x = self.batch_norm(x)
		x = self.activation(x)
		x = self.dropout(x)
		return x


class CBDBlock1d(nn.Module):
	"""
	Block of layers consisting of Conv1d, BatchNorm1d, Activation and Dropout
		Order: Conv1d->BatchNorm1d->Activation->Dropout
	"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=nn.ReLU(), b_norm=True, dropout=False):
		"""
		: param in_channels		Number of input channels (input features).
		: param out_channels	Number of output channels (output features).
		: param activation 		Activation function for this layer (nn.ReLU() <– object type)
		: param dropout         Probability of dropout p. (dropout==False -> no dropout)
		"""
		super(CBDBlock1d, self).__init__()
		self.layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
		self.batch_norm = nn.BatchNorm1d(num_features=out_channels) if b_norm else pass_through
		self.activation = activation
		self.dropout = nn.Dropout(p=dropout) if dropout!=False else pass_through

	def forward(self, X):
		"""
		: param X 			Input data in format (N, C, L) = (N samples in batch, Channels, Length of numbers in channel). (batch of data)
		"""
		X = self.layer(X)
		X = self.batch_norm(X)
		X = self.activation(X)
		X = self.dropout(X)
		return X


class CBDBlockTranspose1d(nn.Module):
	"""
	Block of layers consisting of ConvTramspose1d, BatchNorm1d, Activation and Dropout
		Order: BatchNorm1d->Activation->Dropout->ConvTranspose1
	"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=nn.ReLU(), b_norm=True, dropout=False):
		"""
		: param in_channels		Number of input channels (input features).
		: param out_channels	Number of output channels (output features).
		: param activation 		Activation function for this layer (nn.ReLU() <– object type)
		: param dropout         Probability of dropout p. (dropout==False -> no dropout)
		"""
		super(CBDBlockTranspose1d, self).__init__()
		self.layer = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
		self.batch_norm = nn.BatchNorm1d(num_features=in_channels) if b_norm else pass_through
		self.activation = activation
		self.dropout = nn.Dropout(p=dropout) if dropout!=False else pass_through

	def forward(self, X):
		"""
		: param X 			Input data in format (N, C, L) = (N samples in batch, Channels, Length of numbers in channel). (batch of data)
		"""
		X = self.activation(X)
		X = self.batch_norm(X)
		X = self.dropout(X)
		X = self.layer(X)
		return X


class MixConv1d(nn.Module):
	def __init__(self, in_channels, n_filters, kernel_sizes=[5,11,23], bias=True):
		"""
		MixConv1d layer which combine convolutions with diferent kernel sizes. 
		That helps capture smaller and larger objects (trends) in input data.
		Similiar to InceptionTime layers.

		Params:
			in_channels		Number of input channels (input features).
			n_filters		Number of filters (masks) per convolution.
			kernel_sizes    List of kernel sizes convolutions (example: [3,5] or [4,6,8]...)

		Example:
			>>> from utils.layers import MixConv1d
			>>> m = MixConv1d(3,5,[3,5,7])
			>>> m
			MixConv1d(
			(convolutions): ModuleList(
				(0): Conv1d(3, 5, kernel_size=(3,), stride=(1,), padding=(1,))
				(1): Conv1d(3, 5, kernel_size=(5,), stride=(1,), padding=(2,))
				(2): Conv1d(3, 5, kernel_size=(7,), stride=(1,), padding=(3,))
			)
			)
			>>> x = torch.randn(10,3,20)
			>>> m(x).shape
			torch.Size([10, 15, 20])

		"""
		super(MixConv1d, self).__init__()
		self.convolutions = nn.ModuleList()
		for ks in kernel_sizes:
			self.convolutions.append(
				nn.Conv1d(
					in_channels=in_channels, 
					out_channels=n_filters, 
					kernel_size=ks, 
					padding=ks//2 if ks % 2 else (ks//2-1), 
					bias=bias
				)
			)

	def forward(self, x):
		xs = []
		for conv in self.convolutions:
			xs.append(conv(x))
		xs = torch.cat(xs, axis=1)
		return xs


class MixConvTranspose1d(nn.Module):
	def __init__(self, in_channels, n_filters, kernel_sizes=[5,11,23], bias=True):
		"""
		\"Transposed\" version of MixConv1d.

		Params:
			in_channels		Number of input channels (input features).
			n_filters		Number of filters (masks) per convolution.
			kernel_sizes    List of kernel sizes convolutions (example: [3,5] or [4,6,8]...)

		Example:
			>>> from utils.layers import MixConvTranspose1d
			>>> m = MixConvTranspose1d(3,5,[3,5,7])
			>>> m
			MixConvTranspose1d(
			(convolutions): ModuleList(
				(0): ConvTranspose1d(3, 5, kernel_size=(3,), stride=(1,), padding=(1,))
				(1): ConvTranspose1d(3, 5, kernel_size=(5,), stride=(1,), padding=(2,))
				(2): ConvTranspose1d(3, 5, kernel_size=(7,), stride=(1,), padding=(3,))
			)
			)
			>>> x = torch.randn(10,3,20)
			>>> m(x).shape
			torch.Size([10, 15, 20])

		"""
		super(MixConvTranspose1d, self).__init__()
		self.convolutions = nn.ModuleList()
		for ks in kernel_sizes:
			self.convolutions.append(
				nn.ConvTranspose1d(
					in_channels=in_channels, 
					out_channels=n_filters, 
					kernel_size=ks, 
					padding=ks//2 if ks % 2 else (ks//2-1), 
					bias=bias
				)
			)

	def forward(self, x):
		xs = []
		for conv in self.convolutions:
			xs.append(conv(x))
		xs = torch.cat(xs, axis=1)
		return xs


class ResNetBlock1d(nn.Module):
	def __init__(self, in_channels, out_channels=64, kernel_size=5, layers=2, activation=nn.ReLU):
		super(ResNetBlock1d, self).__init__()
		self.activation = activation()
		self.skip = nn.Conv1d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=1
		) if in_channels!=out_channels else lambda x: x

		block = []
		for i in range(layers):
			block = [
				*block,
				nn.Conv1d(
					in_channels=in_channels, 
					out_channels=out_channels,
					kernel_size=kernel_size,
					stride=1, 
					padding=int(kernel_size//2)
				),
				nn.BatchNorm1d(num_features=out_channels),
			]
			if i < layers-1:
				block.append(activation())
			in_channels=out_channels

		self.block = nn.Sequential(*block)

	def forward(self, x):
		x = self.activation(self.block(x) + self.skip(x))
		return x


class ResNetBlockTranspose1d(nn.Module):
	def __init__(self, in_channels, out_channels=64, kernel_size=5, layers=2, activation=nn.ReLU):
		super(ResNetBlockTranspose1d, self).__init__()
		block = []
		for i in range(layers):
			block = [
				*block,
				nn.ConvTranspose1d(
					in_channels=in_channels, 
					out_channels=out_channels,
					kernel_size=kernel_size,
					stride=1, 
					padding=int(kernel_size//2)
				),
				activation()

			]
		self.block = nn.Sequential(*block)

	def forward(self, x):
		x = self.block(x) + x
		return x


class LSTM_mod(nn.Module):
	"""
	Just regular LSTM layer without returning hidden states and flattened output. 
	This modified layer can be used in nn.Sequential type of model.
	"""
	def __init__(self, sequence_len, **kwargs):
		super(LSTM_mod, self).__init__()
		self.layer = nn.LSTM(**kwargs)
		self.sequence_len = sequence_len
	
	def forward(self,x):
		out, (h,c) = self.layer(x)
		return out.reshape(-1, self.layer.hidden_size*pow(2,int(self.layer.bidirectional))*self.sequence_len)


class LSTM_last_hidden(nn.Module):
	"""
	Regular LSTM layer with added dimension permutation on output. 
	Returns hidden states (h, C) from top (in case of multiple layers) rnn layer.
	"""
	def __init__(self, n_features, hidden_size, num_layers=1):
		super(LSTM_last_hidden, self).__init__()
		self.lstm = nn.LSTM(
			input_size=n_features,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=False,
			bidirectional=False
		)

	def forward(self, x):
		# input to RNN must be [sequence_len, batch_size, n_features] (real input [batch_size, n_features, sequence_len])
		x = x.permute(2, 0, 1) 
		_,(h_end, c_end) = self.lstm(x)
		h_end = h_end[-1, :, :]
		c_end = c_end[-1, :, :]
		return torch.cat((h_end,c_end), axis=-1)


class LSTM_decoder_mod(nn.Module):
	"""
	LSTM layer with modified inputs for use in decoder.
	As input sequence it takes torch.zeros.
	"""
	def __init__(self, n_features, hidden_size, sequence_len, num_layers=1):
		super(LSTM_decoder_mod, self).__init__()
		self.n_features =n_features
		self.sequence_len = sequence_len
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(
			input_size=n_features,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=False,
			bidirectional=False
		)
	
	def forward(self, hidden_state, x=None):
		if (len(hidden_state)==2) and isinstance(hidden_state, tuple):
			h, c = hidden_state
		else: # suppose it is output form dense layer
			h = hidden_state[:, :self.hidden_size].view(1,-1, self.hidden_size).contiguous()
			c = hidden_state[:, self.hidden_size:].view(1,-1, self.hidden_size).contiguous()
		# LSTM layer must take input x even if we don't need it (can't use it)
		x = torch.zeros(self.sequence_len, h.size(1), self.n_features, device=h.device) if x==None else x 
		x, _ = self.lstm(x, (h, c))
		return x


class LSTM_decoder_new(nn.Module):
	def __init__(self, zdim, n_features, hidden_size, sequence_len, activation=nn.Tanh()):
		"""
		Decoder like from seq2seq models, where output from previous step is input to current step.
		There is also added estimation of standard deviation for output sequence (Time and memory expensive).
		"""
		super(LSTM_decoder_new, self).__init__()
		self.n_features = n_features
		self.sequence_len = sequence_len
		self.hidden_size = hidden_size
		self.activation = activation

		self.from_latent = nn.Linear(
			in_features=zdim, 
			out_features=hidden_size*2,
		)
		self.lstm = nn.LSTM(
			input_size=n_features,
			hidden_size=hidden_size,
			batch_first=False,
			bidirectional=False
		)
		self.sigma_from_hidden = nn.Linear(
			in_features=hidden_size*sequence_len,
			out_features=n_features # works for our usage
		)
		self.from_lstm = nn.Linear(
			in_features=hidden_size,
			out_features=n_features
		)
		self.softplus = nn.Softplus()
		self.reshape_sigma = Reshape(out_shape=(n_features, 1))

	def forward(self, z):
		hidden_state = self.from_latent(z)

		h = hidden_state[:, :self.hidden_size].view(1,-1, self.hidden_size).contiguous()
		c = hidden_state[:, self.hidden_size:].view(1,-1, self.hidden_size).contiguous()
		x = torch.zeros(1, h.size(1), self.n_features, device=z.device)

		mu = []#torch.empty(self.sequence_len, h.size(1), self.n_features)
		xs = []#torch.empty(self.sequence_len, h.size(1), self.hidden_size)

		for i in range(self.sequence_len):
			x_, (h,c) = self.lstm(x, (h,c))
			xs.append(x_)
			x = self.from_lstm(self.activation(x_))
			mu.append(x)

		mu = torch.cat(mu, dim=0)
		xs = torch.cat(xs, dim=0)
		mu = mu.permute(1,2,0)
		xs = xs.permute(1,2,0).reshape(-1, self.sequence_len*self.hidden_size)
		sigma = self.softplus(self.sigma_from_hidden(xs))
		sigma = self.reshape_sigma(sigma)

		return mu, sigma
