import sys
sys.path.append("../")

import pandas as pd
import torch 
from tqdm import tqdm

import utils.whole_seq_utils as wsu

#df = pd.read_csv("results/reevaluate.csv")
df = pd.read_csv("results/reevaluate_sota.csv")

dataloaders = wsu.load_and_preprocess_whole_seq(transform=None, batch_size=512, label_idx=80)

models = df.models_saved_as.values

errors = 0
error_models = []
for m in tqdm(models):
	try:
		mod = wsu.Rebuilder(m, True)()
		model = mod.model
		
		model.device=torch.device("cuda")
		model = model.cuda()

		outputs = {}
		model.eval()
		for key in dataloaders.keys():
			y_pred = []
			for x,_ in dataloaders[key]:
				x = x.cuda()
				y_p = model(x).detach().cpu()
				y_pred.append(y_p)
			outputs[key] = torch.vstack(y_pred).numpy()
		
		torch.save(outputs, f"results/whole_reevaluate/{m}")
	except Exception as e:
		errors += 1
		error_models.append([m, e])

pd.set_option('display.max_rows', None)
print(pd.DataFrame(error_models))
print(errors)