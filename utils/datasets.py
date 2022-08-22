import os
import numpy as np 
import torch
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from utils.utils import One_Hot, MultivariateScaler # if path is added to sys else just "from utils import One_Hot"

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


def load_and_preprocess_old(
		mode="sup", transform=None, batch_size=128, validation=True, 
		one_hot=False, sub_samples=1.0, balanced=True, label_idx=80, seed=666
	):
	""" Loading and preprocessing specific data. (tailor-made) """
	assert mode in ["sup", "semisup", "unsup"]
	assert (sub_samples >0.0 and sub_samples <=1.0)
	X_s = []
	X_u = []
	for i in range(10):
		X_s.append(np.load(f"../data/dataset/labeled_sequences_new_{i+1}.npy"))
		X_u.append(np.load(f"../data/dataset/unlabeled_sequences_{i+1}.npy"))


	if label_idx != 80:	
		Y = []	
		for i in range(10):
			Y.append(np.load(f"../data/dataset/label_sequences_new_{i+1}.npy"))
		

	X_s = np.vstack(X_s)
	X_u = np.vstack(X_u)
	y_s = np.vstack(Y) if label_idx != 80 else np.load("../data/dataset/sequence_labels_new.npy")
	# Train test split always the same

	X_train, X_test, X_val, X_u, y_train, y_test, y_val = split_and_scale(X_s, X_u, y_s, test_size=0.2, val_size=0.2, seed=seed)
	
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
			label_idx=label_idx, 
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
			label_idx=label_idx, 
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


def load_and_preprocess_new(
		mode="sup", path="../", transform=None, batch_size=128, validation=True, 
		one_hot=False, sub_samples=1.0, balanced=True, label_idx=80, seed=666
	):
	path = os.path.join(path,"data/dataset/")
	sup_dataset = pd.read_csv(os.path.join(path, "supervised_dataset.csv"))#_sgf
	unsup_dataset = pd.read_csv(os.path.join(path, "unsupervised_dataset.csv"))#_sgf

	X_s, Y, X_u = [], [], []
	for _, seq in sup_dataset.iterrows():
		X_s.append(np.load(os.path.join(path, seq["sequences"])))
		Y.append(np.load(os.path.join(path, seq["labels"])))

	for _, seq in unsup_dataset.iterrows():
		X_u.append(np.load(os.path.join(path, seq["sequences"])))
	
	X_s = np.vstack(X_s)
	X_u = np.vstack(X_u)
	Y = np.vstack(Y)
	y_s = Y[:,label_idx]
	
	vs = 0.2 if validation else None

	X_train, X_test, X_val, X_u, y_train, y_test, y_val = split_and_scale(X_s, X_u, y_s, test_size=0.2, val_size=vs, seed=seed)

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
	
 
def load_and_preprocess_special(
		mode="sup", path="../", transform=None, batch_size=128, validation=True, 
		one_hot=False, sub_samples=1.0, balanced=True, label_idx=80, seed=666
	):
	path = os.path.join(path,"data/dataset/")
	sup_dataset = pd.read_csv(os.path.join(path, "supervised_dataset.csv"))#_sgf
	unsup_dataset = pd.read_csv(os.path.join(path, "unsupervised_dataset.csv"))#_sgf
 
	sup_dataset_sgf = pd.read_csv(os.path.join(path, "supervised_dataset_sgf.csv"))
	unsup_dataset_sgf = pd.read_csv(os.path.join(path, "unsupervised_dataset_sgf.csv"))

	X_s, Xs_sgf, Y, X_u, Xu_sgf = [], [], [], [], []
	for (_, seq), (_, seq_sgf) in zip(sup_dataset.iterrows(),sup_dataset_sgf.iterrows()):
		X_s.append(np.load(os.path.join(path, seq["sequences"])))
		Xs_sgf.append(np.load(os.path.join(path, seq_sgf["sequences"])))
		Y.append(np.load(os.path.join(path, seq["labels"])))
  
	for (_, seq), (_, seq_sgf) in zip(unsup_dataset.iterrows(),unsup_dataset_sgf.iterrows()):
		X_u.append(np.load(os.path.join(path, seq["sequences"])))
		Xu_sgf.append(np.load(os.path.join(path, seq_sgf["sequences"])))
	
	X_s = np.vstack(X_s)
	Xs_sgf = np.vstack(Xs_sgf)
	X_u = np.vstack(X_u)
	Xu_sgf = np.vstack(Xu_sgf)

	X_s = np.hstack((X_s, Xs_sgf[:,1:,:]))
	X_u = np.hstack((X_u, Xu_sgf[:,1:,:]))
	
	Y = np.vstack(Y)
	y_s = Y[:,label_idx]
	
	vs = 0.2 if validation else None

	X_train, X_test, X_val, X_u, y_train, y_test, y_val = split_and_scale(X_s, X_u, y_s, test_size=0.2, val_size=vs, seed=seed)

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
 

def load_and_preprocess_new_extended(
		mode="sup", path="../", transform=None, batch_size=128, validation=True, 
		one_hot=False, sub_samples=1.0, balanced=True, label_idx=80, seed=666
	):
	path = os.path.join(path,"data/dataset/")
	sup_dataset = pd.read_csv(os.path.join(path, "supervised_dataset.csv"))
	unsup_dataset = pd.read_csv(os.path.join(path, "unsupervised_dataset.csv"))

	X_s, Y, X_u = [], [], []
	for _, seq in sup_dataset.iterrows():
		seq_path = seq["sequences"].split(".")[0] + "-allCh.npy"
		X_s.append(np.load(os.path.join(path, seq_path)))
		Y.append(np.load(os.path.join(path, seq["labels"])))

	for _, seq in unsup_dataset.iterrows():
		seq_path = seq["sequences"].split(".")[0] + "-allCh.npy"
		X_u.append(np.load(os.path.join(path, seq_path)))
	
	X_s = np.vstack(X_s)
	X_u = np.vstack(X_u)
	Y = np.vstack(Y)
	y_s = Y[:,label_idx]
	
	vs = 0.2 if validation else None

	X_train, X_test, X_val, X_u, y_train, y_test, y_val = split_and_scale(X_s, X_u, y_s, test_size=0.2, val_size=vs, seed=seed)

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


def load_and_preprocess_whole_seq(path="../", transform=None, batch_size=128, label_idx=80, seed=666):

	path = os.path.join(path,"data/dataset/")
	path_seq = os.path.join(path, "whole_shot/")
	sup_dataset = pd.read_csv(os.path.join(path, "supervised_dataset.csv"))
	unsup_dataset = pd.read_csv(os.path.join(path, "unsupervised_dataset.csv"))

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

	for _, seq in tqdm(sup_dataset.iterrows()):
		shot_number = seq["sequences"].split("-")[-1][:-4]
		seq_path = seq["sequences"].split(".")[0] + "-whole.npy"
		lab_path = seq["labels"].split(".")[0] + "-whole.npy"
		X_dict[shot_number] = np.load(os.path.join(path_seq, seq_path))
		Y_dict[shot_number] = np.load(os.path.join(path_seq, lab_path))
		Y_dict[shot_number] = Y_dict[shot_number][:,label_idx]
		#X_list.append(np.load(os.path.join(path_seq, seq_path)))
		#Y_list.append(np.load(os.path.join(path_seq, lab_path)))

	addi = "18145"
	X_dict[addi] = np.load(os.path.join(path_seq, f"sequences_shot-{addi}-whole.npy"))
	Y_dict[addi] = np.load(os.path.join(path_seq, f"sequences_labels-{addi}-whole.npy"))
	Y_dict[addi] = Y_dict[addi][:,label_idx]
	
	X_train, X_test, y_train, y_test = train_test_split(np.array(X_s), np.array(y_s), test_size=0.2, random_state=seed)

	#scaling
	scaler = MultivariateScaler(dimension=X_train.shape[1])
	scaler.fit(np.vstack((X_train, X_u)))
	
	#print("mean before:", X_dict[list(X_dict.keys())[0]].mean())
	for key in X_dict.keys():
		X_dict[key] = scaler.transform(X_dict[key])
	#print("mean after:", X_dict[list(X_dict.keys())[0]].mean())
	#print(X_dict.keys(), Y_dict.keys())

	dataloaders = {}
	print("building dataloaders")
	for key in tqdm(X_dict.keys()):
		dataset = SupervisedDataset(X_dict[key], Y_dict[key], transform=transform, label_idx=None)
		dataloaders[key] = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

	return dataloaders


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


def load_and_preprocess(mode="sup", transform=None, batch_size=128, validation=True, one_hot=False, sup_samples=1.0, balanced=True, label_idx=80, seed=666):
	""" Loading and preprocessing specific data. (tailor-made) """
	assert mode in ["sup", "semisup", "unsup"]
	assert (sup_samples >0.0 and sup_samples <=1.0)
	X_s = []
	X_u = []
	data_loaders = {}
	for i in range(10):
		X_s.append(np.load(f"../data/dataset/labeled_sequences_new_{i+1}.npy"))
		X_u.append(np.load(f"../data/dataset/unlabeled_sequences_{i+1}.npy"))


	if label_idx != 80:	
		Y = []	
		for i in range(10):
			Y.append(np.load(f"../data/dataset/label_sequences_new_{i+1}.npy"))
		

	X_s = np.vstack(X_s)
	y_s = np.vstack(Y) if label_idx != 80 else np.load("../data/dataset/sequence_labels_new.npy")
	# Train test split always the same
	X_train, X_test, y_train, y_test = model_selection.train_test_split(np.array(X_s), np.array(y_s), test_size=0.2, random_state=666)
	
	# scaling
	scalers = {}
	X_u = np.vstack(X_u)
	# fit scaler
	XX = np.vstack((X_train, X_u)) # scaling both labeled and unlabeled sequences together. reason is further useage
	for i in range(XX.shape[1]):
		scalers[i] = sklearn.preprocessing.RobustScaler()
		scalers[i].fit(XX[:, i, :]) 

	# scale X_train
	for i in range(X_train.shape[1]):
		X_train[:, i, :] = scalers[i].transform(X_train[:, i, :]) 		

	# scale X_u
	for i in range(X_u.shape[1]):
		X_u[:, i, :] = scalers[i].transform(X_u[:, i, :])

	for i in range(X_test.shape[1]):
		X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])
		
	# validation split
	if validation:
		X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
	
	if sup_samples != 1: 
		if balanced:
			indexes = balance_classes_split(y_train, X=None, ratio=sup_samples, seed=seed)
		else:
			n = len(X_train)
			np.random.seed(seed)
			indexes = np.random.choice(np.arange(n), size=int(n*sup_samples), replace=False)
		c_indexes = np.array([i for i in range(len(X_train)) if i not in indexes])

		X_u = np.vstack((X_u, X_train[c_indexes]))
		X_train = X_train[indexes]
		y_train = y_train[indexes]
	
	oh = {True: 4, False: None}

	if label_idx != 80:
		SDataset = SupervisedDatasetMovingLabel
	else:
		SDataset = SupervisedDataset

	# datasets
	if mode == "semisup":
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
		
		if validation:
			val = SDataset(X_validation, y_validation, transform=transform, one_hot=oh[one_hot], label_idx=label_idx) 
			data_loaders["val"] = torch.utils.data.DataLoader(dataset=val, batch_size=batch_size, shuffle=False)
		
		return data_loaders
		
	elif mode == "unsup":
		X = np.vstack((X_train, X_u)) 
		#y = np.hstack((y_train, -1*np.ones(len(X_u)))) 
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
		if validation:
			val = UnsupervisedDataset(X_validation, transform=transform) # y_validation
			data_loaders["val"] = torch.utils.data.DataLoader(dataset=val, batch_size=batch_size, shuffle=False)
			
		return data_loaders
	
	else:
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
		if validation:
			val = SDataset(X_validation, y_validation, transform=transform, label_idx=label_idx)
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

