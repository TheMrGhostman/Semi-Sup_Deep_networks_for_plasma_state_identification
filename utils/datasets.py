import os
import numpy as np 
import torch
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from utils.utils import One_Hot, MultivariateScaler 


def pass_(x):
	return x

def balance_classes_split(y,X=None, ratio=1.0, seed=1):
	"""
	Downsampling number of points to selected ratio while trying to balance classes.
	The biggest class loses most samples.

	It is not uniform downsampling!!!!

	all samples (100) - ratio = 1.0 -> {"apple": 30, "bannana": 40, "pear": 20, "pineapple": 10}
						ratio = 0.9 -> {"apple": 30, "bannana": 30, "pear": 20, "pineapple": 10}
						ratio = 0.8 -> {"apple": 25, "bannana": 25, "pear": 20, "pineapple": 10}
						ratio = 0.5 -> {"apple": 13, "bannana": 13, "pear": 13, "pineapple": 10}

	Params:
		y		Labels
		X		Data tensor (Optional, Default: None). if X is None returns indexes else new X and y
		ratio	Ratio of samples we want to keep.
		seed 	Random seed.
	"""
	def algorithm(arr, nn):
		arr = np.array(arr,dtype=float)
		J1 = np.arange(len(arr))[:-1]
		J2 = np.arange(1,len(arr))

		for i,(j1, j2) in enumerate(zip(J1,J2),1):
			dif = arr[j1] - arr[j2]
			if (dif < nn/i) and (nn > 0):
				arr[:j2+1] = arr[j2]*np.ones_like(arr[:j2+1])
				nn = nn - dif*i
			else:
				arr[:j2] = (arr[j1]-nn/i)*np.ones_like(arr[:j2])
				return arr
		arr = (arr[j1]-nn/(i+1))*np.ones_like(arr)
		return arr
	y_unique = np.unique(y)
	cls_idx = {cl: np.array(y == cl) for cl in y_unique}
	idx = {cl: np.where(np.array(y == cl))[0] for cl in y_unique}
	card = {cl: np.sum(cls_idx[cl]) for cl in y_unique}
	order = {k: v for k, v in sorted(card.items(), key=lambda item: item[1], reverse=True)}.keys()
	
	arr = [card[i] for i in order]
	nn = (1-ratio)*len(y)
	arr = np.round(algorithm(arr, nn)) # number of samples from each class
	print(arr)
	arr = arr[np.array(list(order), dtype=int)] # sorted back 
	np.random.seed(seed)
	indexes = np.hstack([np.random.choice(idx[i], size=int(arr[i]), replace=False) for i in range(len(y_unique))])
	print(f"number of new data / original number = {len(indexes)/len(y)}")
	if np.all(X != None):
		return X[indexes], y[indexes]
	return indexes


def sequence(x, length=160, stride=10):
    tensor = torch.tensor([x[i:i+length] for i in range(0, len(x)-(length-stride+1), stride)])
    tensor = np.moveaxis(tensor.numpy(), [0,1,2], [0,2,1])
    return tensor


def load_unseen_test_data(path="../", batch_size=128, stride=10, downsample=10, seq_len=160, label_idx=80, verbose=True):
	path = os.path.join(path,"data/dataset/")
	unsup_dataset = list(filter(lambda x: ".csv" in x,os.listdir(os.path.join(path, "unsupervised"))))

	if verbose:
		print("Processing unsupervised set")
		unsup_iterator = tqdm(unsup_dataset)
	else:
		unsup_iterator = unsup_dataset

	#if "processed_whole_unsup.pt"
	print("cutting sequences")
	X_dict = {}
	for seq in unsup_iterator:
		_name = seq.split("-")[1].split(".")[0]
		df_tmp = pd.read_csv(os.path.join(path, "unsupervised", seq))
		x = df_tmp[["D_alpha", "IPR1", "IPR9", "IPR14", "McB2"]].values
		x = x[::downsample]
		x = sequence(x, length=seq_len, stride=stride)
		y = df_tmp[["labels"]].values
		y = y[::downsample]
		y = sequence(y, length=seq_len, stride=stride)
		X_dict[_name] = (x, y[:,:,label_idx])

	print("building dataloaders")
	dataloaders = {}
	for key in tqdm(X_dict.keys()):
		dataset = SupervisedDataset(X_dict[key][0], X_dict[key][1])
		dataloaders[key] = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

	return dataloaders


def load_and_preprocess(
		mode="sup", path="../", transform=None, batch_size=128, validation=True, 
		sub_samples=1.0, balanced=True, downsample=10, seq_len=160, stride=10, 
		scale=False, label_idx=80, seed=666, verbose=True
	):
	path = os.path.join(path,"data/dataset/")
	sup_dataset = list(filter(lambda x: ".csv" in x, os.listdir(os.path.join(path, "supervised"))))
	unsup_dataset = list(filter(lambda x: ".csv" in x,os.listdir(os.path.join(path, "unsupervised"))))

	X_s, Y, X_u = [], [], []
	if verbose:
		print("Processing supervised set")
		sup_iterator = tqdm(sup_dataset)
	else:
		sup_iterator = sup_dataset

	for seq in sup_iterator:
		df_tmp = pd.read_csv(os.path.join(path, "supervised", seq))
		x = df_tmp[["D_alpha", "IPR1", "IPR9", "IPR14", "McB2"]].values
		x = x[::downsample]
		x = sequence(x, length=seq_len, stride=stride)
		X_s.append(x)
		y = df_tmp[["labels"]].values
		y = y[::downsample]
		y = sequence(y, length=seq_len, stride=stride)
		Y.append(y)
		
	if verbose:
		print("Processing unsupervised set")
		unsup_iterator = tqdm(unsup_dataset)
	else:
		unsup_iterator = unsup_dataset
	
	for seq in unsup_iterator:
		df_tmp = pd.read_csv(os.path.join(path, "unsupervised", seq))
		x = df_tmp[["D_alpha", "IPR1", "IPR9", "IPR14", "McB2"]].values
		x = x[::downsample]
		x = sequence(x, length=seq_len, stride=stride)
		X_u.append(x)
	
	X_s = np.vstack(X_s)
	X_u = np.vstack(X_u)
	Y = np.vstack(Y)
	y_s = Y[:,0,label_idx]
	
	vs = 0.2 if validation else None

	split_function = split_and_scale if scale else just_split
	X_train, X_test, X_val, X_u, y_train, y_test, y_val = split_function(X_s, X_u, y_s, test_size=0.2, val_size=vs, seed=seed)

	if mode == "sup":
		data_loaders = create_sup_dataloaders(
			X_train=X_train, 
			y_train=y_train, 
			X_test=X_test, 
			y_test=y_test, 
			X_val=X_val, 
			y_val=y_val, 
			transform=transform, 
			batch_size=batch_size, 
			label_idx= None,# if label_idx == 80 else label_idx, 
			balanced=balanced,
			sub_samples=sub_samples,
			seed=seed
		)
		return data_loaders
	elif mode == "semisup":
		data_loaders = create_semisup_dataloaders(
			X_train=X_train, 
			y_train=y_train,
			X_u=X_u, 
			X_test=X_test, 
			y_test=y_test, 
			X_val=X_val, 
			y_val=y_val, 
			transform=transform, 
			batch_size=batch_size,
			one_hot=True, 
			label_idx=None,# if label_idx == 80 else label_idx, 
			balanced=balanced,
			sub_samples=sub_samples,
			seed=seed
		)
		return data_loaders
	else:
		data_loaders = create_unsup_dataloaders(
			X=np.vstack((X_train, X_u)),
			X_test=X_test,  
			X_val=X_val, 
			transform=transform, 
			batch_size=batch_size, 
			seed=seed
		)
		return data_loaders



def split_and_scale(X_s, X_u, y_s, test_size=0.2, val_size=0.2, seed=666):
	# train/test split
	X_train, X_test, y_train, y_test = train_test_split(np.array(X_s), np.array(y_s), test_size=test_size, random_state=seed)

	#scaling
	scaler = MultivariateScaler(dimension=X_train.shape[1])
	scaler.fit(np.vstack((X_train, X_u)))
	X_train = scaler.transform(X_train)
	X_u = scaler.transform(X_u)
	X_test = scaler.transform(X_test)

	if val_size != None:
		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed)
		return X_train, X_test, X_val, X_u, y_train, y_test, y_val
	else:
		return X_train, X_test, None, X_u, y_train, y_test, None


def just_split(X_s, X_u, y_s, test_size=0.2, val_size=0.2, seed=666):
	# train/test split
	X_train, X_test, y_train, y_test = train_test_split(np.array(X_s), np.array(y_s), test_size=test_size, random_state=seed)

	if val_size != None:
		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed)
		return X_train, X_test, X_val, X_u, y_train, y_test, y_val
	else:
		return X_train, X_test, None, X_u, y_train, y_test, None


def subsampling(y, X=None, balanced=True, ratio=1.0, seed=666):
	if balanced:
		indexes = balance_classes_split(y, X=None, ratio=ratio, seed=seed)
	else:
		n = len(y)
		np.random.seed(seed)
		indexes = np.random.choice(np.arange(n), size=int(n*ratio), replace=False)
	c_indexes = np.array([i for i in range(len(y)) if i not in indexes])
	return indexes, c_indexes


def create_sup_dataloaders(
		X_train, y_train, X_test, y_test, X_val=None, y_val=None, 
		transform=None, batch_size=128, label_idx=None, seed=666, 
		balanced=True, sub_samples=1.0, **kwargs
	):
	data_loaders = {}
	if sub_samples != 1.0:
		indexes, _ = subsampling(y_train, balanced=balanced, ratio=sub_samples, seed=seed)
		X_train = X_train[indexes]
		y_train = y_train[indexes]

	if (label_idx != 80) & (label_idx != None):
		SDataset = SupervisedDatasetMovingLabel
	else:
		SDataset = SupervisedDataset

	sup = SDataset(X_train, y_train, transform=transform, label_idx=label_idx)
	test = SDataset(X_test, y_test, transform=transform, label_idx=label_idx)
		
	data_loaders["sup"] = torch.utils.data.DataLoader(
								dataset=sup,
								batch_size=batch_size, 
								shuffle=False, 
								sampler=torch.utils.data.RandomSampler(
									sup, 
									replacement=True
									)
								)
	data_loaders["test"] = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
	if X_val is not None:
		val = SDataset(X_val, y_val, transform=transform, label_idx=label_idx)
		data_loaders["val"] = torch.utils.data.DataLoader(dataset=val, batch_size=batch_size, shuffle=False)
		
	return data_loaders


def create_semisup_dataloaders(
		X_train, y_train, X_u, X_test, y_test, X_val=None, y_val=None, 
		transform=None, batch_size=128, one_hot=False, label_idx=None, 
		balanced=True, sub_samples=1.0, seed=666, **kwargs
	):

	data_loaders = {}
	oh = {True: len(np.unique(y_train)), False: None}

	if sub_samples != 1:
		indexes, c_indexes = subsampling(y_train, balanced=balanced, ratio=sub_samples, seed=seed)
		X_u = np.vstack((X_u, X_train[c_indexes]))
		X_train = X_train[indexes]
		y_train = y_train[indexes]

	if (label_idx != 80) & (label_idx != None):
		SDataset = SupervisedDatasetMovingLabel
	else:
		SDataset = SupervisedDataset

	sup = SDataset(X_train, y_train, transform=transform, one_hot=oh[one_hot], label_idx=label_idx)
	test = SDataset(X_test, y_test, transform=transform, one_hot=oh[one_hot], label_idx=label_idx)
		
	data_loaders["sup"] = torch.utils.data.DataLoader(
								dataset=sup,
								batch_size=batch_size, 
								shuffle=False, 
								sampler=torch.utils.data.RandomSampler(
									sup, 
									replacement=True,
									num_samples=X_u.shape[0]
									)
								)
	data_loaders["unsup"] = torch.utils.data.DataLoader(
								dataset=DummyDataset(
									torch.tensor(X_u).float(),
									transform=transform
									), 
								batch_size=batch_size, 
								shuffle=False, 
								sampler=torch.utils.data.RandomSampler(
									X_u, 
									replacement=False
									)
								)

	data_loaders["test"] = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
	
	if X_val is not None:
		val = SDataset(X_val, y_val, transform=transform, one_hot=oh[one_hot], label_idx=label_idx) 
		data_loaders["val"] = torch.utils.data.DataLoader(dataset=val, batch_size=batch_size, shuffle=False)
	
	return data_loaders


def create_unsup_dataloaders(X, X_test, X_val=None, transform=None, batch_size=128, seed=666, **kwargs):
	data_loaders = {}

	unsup = UnsupervisedDataset(X, transform=transform)
	
	data_loaders["unsup"] = torch.utils.data.DataLoader(
								dataset=unsup, 
								batch_size=batch_size, 
								shuffle=False, 
								sampler=torch.utils.data.RandomSampler(
									unsup, 
									replacement=False
									)
								)
	
	test = UnsupervisedDataset(X_test, transform=transform) # y_test
	data_loaders["test"] = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
	if X_val is not None:
		val = UnsupervisedDataset(X_val, transform=transform) # y_validation
		data_loaders["val"] = torch.utils.data.DataLoader(dataset=val, batch_size=batch_size, shuffle=False)
		
	return data_loaders


class SupervisedDataset(torch.utils.data.Dataset):
	def __init__(self, dataset, labels, transform=None, one_hot=None, **kwargs):
		self.X = torch.tensor(dataset.astype('float32')) if isinstance(dataset, np.ndarray) else dataset.float()
		self.y = torch.tensor(labels).long() if isinstance(labels, np.ndarray) else labels.float()
		self.transform = transform
		self.one_hot = (lambda x: x) if one_hot==None else One_Hot(one_hot)

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample_X = self.X[idx] 
		sample_y = self.y[idx]

		if self.transform:
			sample_X = self.transform(sample_X)

		return sample_X, self.one_hot(sample_y)


class UnsupervisedDataset(torch.utils.data.Dataset):
	def __init__(self, X, y=None, transform=None):
		self.X = torch.tensor(X.astype('float32')) if isinstance(X, np.ndarray) else X.float()
		if y==None:
			self.y = self.X.clone().detach()
		else:
			self.y = torch.tensor(y.astype('float32')) if isinstance(y, np.ndarray) else y.float()
		self.transform = transform

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample_X = self.X[idx] 
		sample_y = self.y[idx]

		if self.transform:
			sample_X = self.transform(sample_X)
			sample_y = self.transform(sample_y)# needed for only H_a training

		return sample_X, sample_y


class DummyDataset(torch.utils.data.Dataset):
	def __init__(self, X, transform=None):
		self.X = torch.tensor(X.astype('float32')) if isinstance(X, np.ndarray) else X.float()
		self.transform = transform

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample_X = self.X[idx] 

		if self.transform:
			sample_X = self.transform(sample_X)
		return sample_X


class SupervisedDatasetMovingLabel(torch.utils.data.Dataset):
	def __init__(self, dataset, labels, label_idx=None, transform=None, one_hot=None):
		"""
		Supervised dataset for sequences. 
		
		X are sequences (N, channals, sequence_len)
		y are 1D sequences (N, sequence_len)

		if \"label_idx\" is not None => returns y[idx, label_idx]
		"""
		self.X = torch.tensor(dataset.astype('float32')) if isinstance(dataset, np.ndarray) else dataset.float()
		self.y = torch.tensor(labels.astype("float32")) if isinstance(labels, np.ndarray) else labels.float() 
		self.label_idx = label_idx
		self.transform = transform
		self.one_hot = (lambda x: x) if one_hot==None else One_Hot(one_hot)

	def __len__(self):
		return self.X.shape[0]
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample_X = self.X[idx] 
		sample_y = self.y[idx,self.label_idx].long() if self.label_idx != None else self.y[idx]

		if self.transform:
			sample_X = self.transform(sample_X)

		return sample_X, self.one_hot(sample_y)
