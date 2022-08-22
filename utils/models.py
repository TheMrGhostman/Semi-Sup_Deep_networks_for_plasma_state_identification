import time
import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

from .layers import *
from .losses import Gaussian_NLL
from .utils import compute_dims_1d, correct_sizes, get_activation
from .inception import Inception, InceptionBlock

from IPython.core.debugger import set_trace


# Models
class VAE(nn.Module):
	def __init__(self, encoder, decoder):
		super(VAE, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x):
		z, mu, sigma = self.encoder(x)
		return self.decoder(z), mu, sigma

	def elbo(self, x, y, loss_f=Gaussian_NLL(), samples=1):
		"""
		Evidence Lower Bound (ELBO) for Variational Autoencoder
		log p(x) = E[ log p(x|z) ] - KLD( q(z|x) || p(z) )

		:param x 			- 3D or 2D input tensor . 4D is not yet supported! 
		:param y			- 3D or 2D desired output tensor.
		:param loss_f		- NegLogLikelihood function or class for computation of reconstruction error. 
		:param samples		- Number of samples from latent space.
		
		"""
		_, mu, sigma = self.encoder(x)
		kld = - 0.5 * torch.mean(torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2), axis=1))

		likelihood = 0  
		for i in range(samples):
			z = mu + sigma*torch.randn_like(sigma)
			likelihood -=  loss_f(self.decoder(z), y) #-(-log(p_x)) <= loss_f is negative log(p_x)

		return likelihood/samples - kld


class M1(nn.Module):
	def __init__(self, encoder, classifier, freeze=True, sampling=False):
		super(M1, self).__init__()
		self.encoder = encoder
		self.classifier = classifier
		self.sampling = sampling
		self.frozen = False
		if freeze:
			self.freeze_encoder()

	def freeze_encoder(self):
		for param in self.encoder.parameters():
			param.requires_grad = False
		self.frozen = True
		print("Encoder frozen.")

	def unfreeze_encoder(self):
		for param in self.encoder.parameters():
			param.requires_grad = True
		self.frozen = False
		print("Encoder unfrozen.")

	def forward(self, X):
		Z, mu, sigma = self.encoder(X)
		if self.sampling == True:
			y = self.classifier(Z)
		else:
			y = self.classifier(mu)
		return y


# Constructors for different kinds of VAEs

def cvae(xdim=[5,160], zdim=16, n_filters=[16,32,64], fsizes=[6,4,4], stride=1, padding=0,
	batch_norm=True, dense=None, activation=nn.ReLU, dropout=False, ydim=0, q_zxy=False, **kwargs):

	"""
	Params:
		xdim			Dimensions of single sample. 
		zdim			Dimension of latent space
		n_filters		Number of input channels for hidden conv layers.
		fsizes			Size of feature masks in hidden conv layers.
		stride			Stride for all conv layers.
		padding			Padding for all conv layers
		batch_norm		Logical if use BatchNorm1d or not.
		dense			Number of neurons in Dense layer. If \"None\" -> no hidden dense layer.		
		activation 		Activation function for this layer (nn.ReLU <– object type)
		dropout         Probability of dropout p. (dropout==False -> no dropout)

	Optional parmeters for semi-supervised vae:
		ydim			Number of classes (Default: 0)
		q_zxy			Type of encoder (Default: False). If False => q_zx. 
	"""

	encoder = []
	decoder = []

	conv_dim, _ = compute_dims_1d(
		L_in=xdim[1], 
		ksizes=fsizes, 
		strides=stride, 
		padding=padding, 
		check_for_error=True
	)
	from_conv_dim = conv_dim[-1]*n_filters[-1]
	print(f"Lengths of input tensor after all conv layers: {conv_dim}")

	# Encoder
	in_channels = copy.copy(n_filters[:-1])
	in_channels.insert(0, xdim[0])

	for (i, k, o) in zip(in_channels, fsizes, n_filters):
		encoder.append(
			CBDBlock1d(
				in_channels=i, 
				out_channels=o, 
				kernel_size=k, 
				stride=stride, 
				padding=padding,
				bias= not batch_norm, 
				activation=activation(), 
				b_norm=batch_norm,
				dropout=dropout
			)
		)
	
	encoder.append(Flatten(out_features=from_conv_dim))

	if dense != None:
		encoder.append(nn.Linear(in_features=from_conv_dim + ydim*int(q_zxy), out_features=dense))
		encoder.append(activation())
		encoder.append(VariationalLayer(in_features=dense, out_features=zdim, return_KL=False))
	else:
		encoder.append(VariationalLayer(in_features=from_conv_dim + ydim*int(q_zxy), out_features=zdim, return_KL=False))

	# Decoder
	in_channels.reverse()
	fsizes.reverse()
	n_filters.reverse()

	if dense != None:
		decoder.append(nn.Linear(in_features=zdim + ydim, out_features=dense))
		decoder.append(activation())
		decoder.append(nn.Linear(in_features=dense, out_features=from_conv_dim))
	else:
		decoder.append(nn.Linear(in_features=zdim + ydim, out_features=from_conv_dim))

	decoder.append(Reshape(out_shape=(n_filters[0], int(from_conv_dim / n_filters[0]))))

	for (i, k, o) in zip(n_filters[:-1], fsizes[:-1], in_channels[:-1]):
		decoder.append(
			CBDBlockTranspose1d(
				in_channels=i, 
				out_channels=o, 
				kernel_size=k, 
				stride=stride, 
				padding=padding,
				bias= not batch_norm, 
				activation=activation(), 
				b_norm=batch_norm,
				dropout=dropout
			)
		)

	if batch_norm:
		decoder.append(nn.BatchNorm1d(num_features=n_filters[-1]))
	decoder.append(activation())
	decoder.append(
		ConvDecoderOutput(
			in_channels=n_filters[-1],
			in_features=conv_dim[1], # dim after first conv layer from encoder
			out_channels=in_channels[-1],
			kernel_size=fsizes[-1],
			stride=stride,
			bias=True
		)
	)

	encoder = nn.Sequential(*encoder)
	decoder = nn.Sequential(*decoder)

	return VAE(encoder, decoder)


def cvae2(xdim=[5,160], zdim=16, n_filters=[16,32,64], fsizes=[6,4,4], stride=[1,1,1], padding=[0,0,0], 
	pooling=None, dense=None, activation=nn.ReLU, ydim=0, q_zxy=False, upsample="conv", **kwargs):

	assert upsample in ["conv", "interpolate"]
	encoder = []
	decoder = []

	conv_dim, _ = compute_dims_1d(
		L_in=xdim[1], 
		ksizes=fsizes, 
		strides=stride, 
		padding=padding,
		pooling=pooling if pooling!=None else 1,
		check_for_error=True
	)
	from_conv_dim = conv_dim[-1]*n_filters[-1]
	print(f"Lengths of input tensor after all conv layers: {conv_dim}")

	# Encoder
	in_channels = copy.copy(n_filters[:-1])
	in_channels.insert(0, xdim[0])

	if (pooling!=None) & isinstance(pooling, int):            
		pooling = len(padding)*[pooling]
	if pooling==None:
		pooling = len(padding)*[None]

	for (i, k, o, s, p, pool) in zip(in_channels, fsizes, n_filters, stride, padding, pooling):
		encoder.append(nn.Conv1d(in_channels=i, out_channels=o, kernel_size=k, stride=s, padding=p))
		encoder.append(activation())
		if (pool!=None) & (pool!=0):
			encoder.append(nn.MaxPool1d(pool))

	encoder.append(Flatten(out_features=from_conv_dim))
	if dense != None:
		encoder.append(nn.Linear(in_features=from_conv_dim + ydim*int(q_zxy), out_features=dense))
		encoder.append(activation())
		encoder.append(VariationalLayer(in_features=dense, out_features=zdim, return_KL=False))
	else:
		encoder.append(VariationalLayer(in_features=from_conv_dim + ydim*int(q_zxy), out_features=zdim, return_KL=False))

	# Decoder
	in_channels.reverse()
	fsizes.reverse()
	n_filters.reverse()
	stride.reverse()
	padding.reverse()
	pooling.reverse()

	if dense != None:
		decoder.append(nn.Linear(in_features=zdim + ydim, out_features=dense))
		decoder.append(activation())
		decoder.append(nn.Linear(in_features=dense, out_features=from_conv_dim))
	else:
		decoder.append(nn.Linear(in_features=zdim + ydim, out_features=from_conv_dim))

	decoder.append(Reshape(out_shape=(n_filters[0], int(from_conv_dim / n_filters[0]))))

	for (i, k, o, s, p, pool) in zip(n_filters[:-1], fsizes[:-1], in_channels[:-1], stride[:-1], padding[:-1], pooling[:-1]):
		if (pool!=None) & (pool!=0):
			if upsample=="conv":
				decoder.append(
					nn.ConvTranspose1d(in_channels=n_filters[0], out_channels=i, kernel_size=pool, stride=pool)
				)
			elif upsample=="interpolate":
				decoder.append(
					nn.Upsample(scale_factor=pool, mode="linear")
				)
		#decoder.append(activation())
		decoder.append(nn.ConvTranspose1d(in_channels=i, out_channels=o, kernel_size=k, stride=s, padding=p))
		decoder.append(activation())
		
	if (pooling[-1]!=None) & (pooling[-1]!=0):
		if upsample=="conv":
			decoder.append(
				nn.ConvTranspose1d(in_channels=n_filters[-1], out_channels=n_filters[-1], kernel_size=pooling[-1], stride=pooling[-1])
			)
			decoder.append(activation())
		elif upsample=="interpolate":
			decoder.append(
				nn.Upsample(scale_factor=pooling[-1], mode="linear")
			)
			decoder.append(activation())

	if (pooling==None) | (pooling[0]==None):
		in_feat = conv_dim[1]
	elif sum(pooling)==0:
		in_feat = conv_dim[1]
	else:
		in_feat = conv_dim[0]

	
	decoder.append(
		ConvDecoderOutput(
			in_channels=n_filters[-1],
			in_features=in_feat, 
			out_channels=in_channels[-1],
			kernel_size=fsizes[-1],
			stride=stride[-1],
			padding=padding[-1]
		)
	)

	encoder = nn.Sequential(*encoder)
	decoder = nn.Sequential(*decoder)

	return VAE(encoder, decoder)


def mixconv_vae(xdim=[5,160], zdim=16, n_filters=[32,32,32], mixconv_fsizes=[5,11,23],  
	batch_norm=True, dense=None, activation=nn.ReLU, ydim=0, q_zxy=False, **kwargs):
	"""
	Params:
		xdim			Dimensions of single sample. 
		zdim			Dimension of latent space
		n_filters		Number of input channels for hidden conv layers.
		mixconv_fsizes	Size of feature masks in mixconv layer (one layer have multiple kernel sizes).
						Must be odd numbers or will be corrected!!
		batch_norm		Logical if use BatchNorm1d or not.	
		dense			Number of neurons in Dense layer. If \"None\" -> no hidden dense layer.	
		activation 		Activation function for this layer (nn.ReLU <– object type).
		
	Optional parmeters for semi-supervised vae:
		ydim			Number of classes (Default: 0)
		q_zxy			Type of encoder (Default: False). If False => q_zx. 
	"""
	encoder = []
	decoder = []

	mixconv_fsizes = correct_sizes(mixconv_fsizes)
	print(mixconv_fsizes)
	
	conv_dim, _ = compute_dims_1d(
		L_in=xdim[1], 
		ksizes=[mixconv_fsizes[0]]*len(n_filters), 
		strides=1, 
		padding=mixconv_fsizes[0]//2,
		check_for_error=True
	)
	from_conv_dim = conv_dim[-1]*n_filters[-1]
	print(f"Lengths of input tensor after all conv layers: {conv_dim}")

	# Encoder
	in_channels = copy.copy(n_filters[:-1])
	in_channels.insert(0, xdim[0])
	multiplier = len(mixconv_fsizes) * np.ones(len(n_filters), dtype=int)
	multiplier[0] = 1
	
	for (i, m, o) in zip(in_channels, multiplier, n_filters):
		encoder.append(
			MixConv1d(
				in_channels=i*m, 
				n_filters=o, 
				kernel_sizes=mixconv_fsizes, 
				bias=not batch_norm
			)
		)
		if batch_norm:
			encoder.append(nn.BatchNorm1d(num_features=o*len(mixconv_fsizes)))
		encoder.append(
			activation()
		)

	encoder.append(Flatten(out_features=from_conv_dim*len(mixconv_fsizes)))

	in_features_ = from_conv_dim*len(mixconv_fsizes) + ydim*int(q_zxy)

	if dense != None:
		encoder.append(nn.Linear(in_features=in_features_, out_features=dense))
		encoder.append(activation())
		encoder.append(VariationalLayer(in_features=dense, out_features=zdim, return_KL=False))
	else:
		encoder.append(VariationalLayer(in_features=in_features_, out_features=zdim, return_KL=False))

	# Decoder
	in_channels.reverse()
	n_filters.reverse()

	if dense != None:
		decoder.append(nn.Linear(in_features=zdim + ydim, out_features=dense))
		decoder.append(activation())
		decoder.append(nn.Linear(in_features=dense, out_features=from_conv_dim))
	else:
		decoder.append(nn.Linear(in_features=zdim + ydim, out_features=from_conv_dim))

	decoder.append(Reshape(out_shape=(n_filters[0], int(from_conv_dim / n_filters[0]))))

	for (i, m, o) in zip(n_filters, multiplier, in_channels):
		decoder.append(activation())
		if batch_norm:
			decoder.append(nn.BatchNorm1d(num_features=i*m))
		decoder.append(
			MixConvTranspose1d(
				in_channels=i*m,
				n_filters=o,
				kernel_sizes=mixconv_fsizes,
				bias=not batch_norm
			)
		)
	if batch_norm:
		decoder.append(nn.BatchNorm1d(num_features=xdim[0]*len(mixconv_fsizes)))
	decoder.append(activation())
	decoder.append(
		ConvDecoderOutput(
			in_channels=xdim[0]*len(mixconv_fsizes),
			in_features=conv_dim[1], 
			out_channels=in_channels[-1],
			kernel_size=1,
			bias=True
		)
	)
	encoder = nn.Sequential(*encoder)
	decoder = nn.Sequential(*decoder)

	return VAE(encoder, decoder)


def vae(xdim=[5,160], zdim=16, hidden_neurons=[16,32,64], batch_norm=True, 
	activation=nn.ReLU, dropout=False, ydim=0, q_zxy=False, **kwargs):
	"""
	Params:
		xdim				Dimensions of single sample. 
		zdim				Dimension of latent space
		hidden_neurons		Number of neurons in hidden layers (encoder and decoder are symetric).
		batch_norm			Logical if use BatchNorm1d or not.	
		activation 			Activation function for all layers (nn.ReLU() <– object type)
		dropout        		Probability of dropout p. (dropout==False -> no dropout)

	Optional parmeters for semi-supervised vae:
		ydim			Number of classes (Default: 0)
		q_zxy			Type of encoder (Default: False). If False => q_zx.
	"""
	encoder = []
	decoder = []

	x_in_dim = np.prod(xdim)
	in_features = copy.copy(hidden_neurons[:-1])
	in_features.insert(0, x_in_dim + ydim*int(q_zxy))

	encoder.append(Flatten(out_features=x_in_dim)) # Flatten 3D tensor to 2D

	for (i, o) in zip(in_features, hidden_neurons):
		encoder.append(
			DenseBlock(
				input_dim=i,
				output_dim=o,
				activation=activation(),
				batch_norm=batch_norm,
				dropout=dropout
			)
		)

	encoder.append(VariationalLayer(in_features=hidden_neurons[-1], out_features=zdim, return_KL=False))

	in_features.reverse()
	hidden_neurons.reverse()
	hidden_neurons.insert(0, zdim + ydim)
	in_features.insert(0, hidden_neurons[1])

	for (i, o) in zip( hidden_neurons[:-1], in_features[:-1]):
		decoder.append(
			DenseBlock(
				input_dim=i,
				output_dim=o,
				activation=activation(),
				batch_norm=batch_norm,
				dropout=dropout
			)
		)
	
	decoder.append(
		VariationalDecoderOutput(
			in_features=hidden_neurons[-1], 
			mu_out=x_in_dim, 
			sigma_out=xdim[0], 
			bias=True, 
			reshape=True
		)
	)

	encoder = nn.Sequential(*encoder)
	decoder = nn.Sequential(*decoder)

	return VAE(encoder, decoder)


def rvae_lstm(xdim=[5,160], zdim=16, hidden_size=128, n_layers=1, activation=nn.ReLU, **kwargs):
	"""
	Params:
		xdim				Dimensions of single sample. 
		zdim				Dimension of latent space
		hidden_size			Number of neurons in lstm layers.
		activation 			Activation function for rnn output of decoder (nn.ReLU() <– object type)
	"""
	decoder = []

	encoder = nn.Sequential(
		LSTM_last_hidden(
			n_features=xdim[0], 
			hidden_size=hidden_size, 
			num_layers=n_layers # experimental # default is 1 recurrent layer
		),
		VariationalLayer(
			in_features=hidden_size*2, 
			out_features=zdim,
			return_KL=False
		)
	)

	decoder = nn.Sequential(
		nn.Linear(
			in_features=zdim, 
			out_features=hidden_size*2,
		),
		LSTM_decoder_mod(
			n_features=xdim[0],
			hidden_size=hidden_size,
			sequence_len=xdim[1],
			num_layers=1
		),
		activation(),
		RecurrentDecoderOutput(
			in_features=hidden_size,
			sequence_len=xdim[1],
			out_features=xdim[0]
		)
	)

	return VAE(encoder, decoder)


def rvae_lstm_new(xdim=[5,160], zdim=16, hidden_size=128, activation=nn.ReLU, **kwargs):
	"""
	Params:
		xdim				Dimensions of single sample. 
		zdim				Dimension of latent space
		hidden_size			Number of neurons in lstm layers.
		activation 			Activation function for rnn output of decoder (nn.ReLU() <– object type)
	"""
	decoder = []

	encoder = nn.Sequential(
		LSTM_last_hidden(
			n_features=xdim[0], 
			hidden_size=hidden_size, 
			num_layers=1
		),
		VariationalLayer(
			in_features=hidden_size*2, 
			out_features=zdim,
			return_KL=False
		)
	)

	decoder = LSTM_decoder_new(
		zdim= zdim, 
		n_features=xdim[0], 
		hidden_size=hidden_size, 
		sequence_len=xdim[1], 
		activation=activation()
	)

	return VAE(encoder, decoder)


def inception_time(xdim=5, ydim=4, n_filters=32, fsizes=[5,11,23], bottleneck=32, blocks=1, activation="relu", **kwargs):
	fsizes = correct_sizes(fsizes)

	inceptionBlocks = []
	channels = [xdim] + [n_filters*4*1]*(blocks-1)
	for i in range(blocks):
		inceptionBlocks.append(
			InceptionBlock(
					in_channels=channels[i], 
					n_filters=n_filters, 
					kernel_sizes=fsizes,
					bottleneck_channels=bottleneck,
					use_residual=True,
					activation=get_activation(activation)()
				)
		)

	IT = nn.Sequential(
				*inceptionBlocks,
				nn.AdaptiveAvgPool1d(output_size=1),
				Flatten(out_features=n_filters*4*1),
				nn.Linear(in_features=4*n_filters*1, out_features=ydim)
	)
	return IT