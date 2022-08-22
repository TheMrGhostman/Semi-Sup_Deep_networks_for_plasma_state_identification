import sys
sys.path.append("../")


import argparse
import time
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score

from utils.datasets import load_and_preprocess_new

import matplotlib.pyplot as plt
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="size of each image batch")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--sub_samples", type=float, default=1.0, help="ratio of labeled date")
parser.add_argument("--seed", type=int, default=666, help="validation split seed")

options = parser.parse_args()
print(options)


data_loaders = load_and_preprocess_new(
	mode="sup", 
	batch_size=options.batch_size, 
	validation=True, 
	sub_samples=options.sub_samples, 
	balanced=True,
	seed=options.seed
)

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


crnn = CRNN(torch.device("cuda")).cuda()
opt = torch.optim.Adam(crnn.parameters(), lr=0.0001)
model_nm = f"SOTA_{np.round(time.time())}_lr={options.lr}_sub-sampes={options.sub_samples}_bs={options.batch_size}_seed={options.seed}"

#model_nm +="_deriv=sgf"

losses = []

crnn.train()
for i in tqdm(range(options.epochs)):
    st = time.time()
    for X,y in data_loaders["sup"]:
        X = X.cuda()
        y = y.cuda()
        loss = nn.CrossEntropyLoss()(crnn(X),y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
print("end")
torch.save(crnn.state_dict(), f"checkpoints/{model_nm}.pt")


crnn.eval()
ys =[]
y_hats = []
for x,y in data_loaders["val"]:
    x = x.cuda()
    y_hat = np.argmax(crnn(x).detach().cpu().numpy(), axis=1)
    ys.append(y)
    y_hats.append(y_hat)
f1_v = f1_score(y_true=np.hstack(ys), y_pred=np.hstack(y_hats), average="macro")
ac_v = accuracy_score(y_true=np.hstack(ys), y_pred=np.hstack(y_hats))

crnn.eval()
ys =[]
y_hats = []
for x,y in data_loaders["test"]:
    x = x.cuda()
    y_hat = np.argmax(crnn(x).detach().cpu().numpy(), axis=1)
    ys.append(y)
    y_hats.append(y_hat)

f1 = f1_score(y_true=np.hstack(ys), y_pred=np.hstack(y_hats), average="macro")
ck = cohen_kappa_score(y1=np.hstack(ys), y2=np.hstack(y_hats))
ac = accuracy_score(y_true=np.hstack(ys), y_pred=np.hstack(y_hats))

history = {"loss": losses, "f1_test": f1, "accuracy_test": ac, "f1_val": f1_v, "accuracy_val": ac_v, "cohen_kappa": ck}

torch.save(history, f"model_histories/sota/{model_nm}.pt")