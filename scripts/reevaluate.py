import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import torch 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from scipy.signal import convolve2d

from itertools import chain
from utils.ts_metrics import LH_precision, HElm_precision, LH_recall, HElm_recall, Precision_T, get_ranges
import utils.whole_seq_utils as wsu

def equal_to(X,Y):
	return [x in Y for x in X]

#5 (1) x, 50 (10) , 100 (20) , 150 (30) , 200 (40) , 250 (50) ---- us (timeframe)

SMOOTHING_FACTOR = int(sys.argv[1])#20
MASK = 1 if len(sys.argv) == 2 else int(sys.argv[2])

CALIBRATE = f"conv_smoothing-f{SMOOTHING_FACTOR}"
SMOOTH = f"smooth{SMOOTHING_FACTOR}"

print(f"calibration: {CALIBRATE} | smoothing : {SMOOTH} | mask : {MASK}")

SIMPLE = False
#f = lambda x: x
#smoothing 
mask = (1/SMOOTHING_FACTOR)*np.ones((SMOOTHING_FACTOR,1))

f = lambda x: convolve2d(np.array(x), mask)[:x.shape[0],:]

Y_dict = wsu.get_Y_dict(label_loc=80)
df = pd.read_csv("results/models.csv") #list of trained models with basic parameters etc

columns = [
	'best_loss', 'models_saved_as', 'parameters', 'model_name', 
	'model_type', 'encoder-idx', 'sub_samples', 
	 'es_trace'
	]

# names of models we want to compute all transition metrics
# example
nms = [
'21-07-2021--03-17-28--SSVAE-InceptionTime_lr=0.0005_bs=64_zdim=15_bneck=64_blocks=2_nfilters=32_fsize=5-11-23_activation=swish_scheduler=None'
]

df = df[equal_to(df.model_name, nms)]


df_new = []
errors = 0
error_models = []
ii = 0
for m in tqdm(nms): # df.models_saved_as.values # df.models_saved_as.values
	try:
		Y_pred = torch.load(f"results/{m}") # predictions from model "m" aare saved in results as dict where key is number of discharge/shot (example Y_pred["9749"] are predictions for shot #9749)
		df_model = []
		for key in Y_dict.keys():
			y_pred = f(Y_pred[key]) # prepaation for calibration
			#TODO add smoothing prediction
			y_pred = np.argmax(np.array(y_pred), axis=1)
			y_true = np.array(Y_dict[key])

			prec = precision_score(y_true=y_true, y_pred=y_pred, average=None)
			f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
			acc = accuracy_score(y_true=y_true, y_pred=y_pred)

			if SIMPLE:
				info = {
					"model": m,
					"shot": key,
					"calibration": CALIBRATE, 
					"Acc": acc,
					"F1": f1
				}

			else:
				info = {
					"model": m,
					"shot": key,
					"calibration": CALIBRATE, 
					"Acc": acc,
					"F1": f1,
					"H_precision_standard": prec[0],
					"L_precision_standard": prec[1],
					"Es_precision_standard": prec[2],
					"Ee_precision_standard": prec[3],
					"Mean_precision_standard": np.mean(prec),
				}

				transitions = []
				for lh_tau in chain(range(10,100,10), range(100,2000,100)):
					LH_pr, N_pr = LH_precision(y_pred=y_pred, y_true=y_true, tau=lh_tau, mask=MASK, aggregation="pre-mean")
					LH_re, N_re = LH_recall(y_pred=y_pred, y_true=y_true, tau=lh_tau, mask=MASK, aggregation="pre-mean")

					try:
						LH_F1 = 2*(LH_pr/N_pr)*(LH_re/N_re)/((LH_pr//N_pr) + (LH_re/N_re))
					except (TypeError, ZeroDivisionError):
						LH_F1 = 0

					transitions.append({f"LH_precision-{lh_tau}":LH_pr, f"LH_recall-{lh_tau}":LH_re, f"LH_F1-{lh_tau}":LH_F1, f"LH_N_precision-{lh_tau}":N_pr, f"LH_N_recall-{lh_tau}":N_re})

				for i in transitions:
					info = {**info, **i}

				# left-sided
				transitions = []
				for he_tau in chain(range(1,10), range(10,100,10)):
					HE_pr, N_pr = HElm_precision(y_pred=y_pred, y_true=y_true, tau=(0, he_tau), mask=MASK, aggregation="pre-mean")
					HE_re, N_re = HElm_recall(y_pred=y_pred, y_true=y_true, tau=(0, he_tau), mask=MASK, aggregation="pre-mean")

					try:
						HE_F1 = 2*(HE_pr/N_pr)*(HE_re/N_re)/((HE_pr/N_pr) + (HE_re/N_re))
					except (TypeError, ZeroDivisionError):
						HE_F1 =0

					transitions.append({f"HE_precision-0-{he_tau}":HE_pr, f"HE_recall-0-{he_tau}":HE_re, f"HE_F1-0-{he_tau}":HE_F1, f"HE_N_precision-0-{he_tau}":N_pr, f"HE_N_recall-0-{he_tau}":N_re,})

				for i in transitions:
					info = {**info, **i}
	 
				# right-sided
				transitions = []
				for he_tau in chain(range(1,10), range(10,100,10)):
					HE_pr, N_pr = HElm_precision(y_pred=y_pred, y_true=y_true, tau=(he_tau,0), mask=MASK, aggregation="pre-mean")
					HE_re, N_re = HElm_recall(y_pred=y_pred, y_true=y_true, tau=(he_tau,0), mask=MASK, aggregation="pre-mean")

					try:
						HE_F1 = 2*(HE_pr/N_pr)*(HE_re/N_re)/((HE_pr/N_pr) + (HE_re/N_re))
					except (TypeError, ZeroDivisionError):
						HE_F1 =0

					transitions.append({f"HE_precision-{he_tau}-0":HE_pr, f"HE_recall-{he_tau}-0":HE_re, f"HE_F1-{he_tau}-0":HE_F1, f"HE_N_precision-{he_tau}-0":N_pr, f"HE_N_recall-{he_tau}-0":N_re,})

				for i in transitions:
					info = {**info, **i}

			df_model.append(info)

		df_model = pd.DataFrame(df_model)
		# first save default columns
		def_cols = ["model", "calibration"]
		new_df_cols = [*def_cols]
		new_df = [*df_model[def_cols].iloc[0]]
		#  cols for average
		if SIMPLE:
			avg_cols = ["Acc", "F1"]
		else:
			avg_cols = [
				'Acc', 
				'F1', 
				'H_precision_standard',
				'L_precision_standard',
				'Es_precision_standard',
				'Ee_precision_standard',
				'Mean_precision_standard'
			]
		avgs = df_model[avg_cols].mean().values
		new_df_cols = [*new_df_cols, *avg_cols]
		new_df = [*new_df, *avgs]
		# cols to sum
		# first drop already used or not needed columns
		tmp = df_model.drop(columns=avg_cols)
		tmp = tmp.drop(columns=[c for c in tmp.columns if "F1" in c])
		tmp = tmp.drop(columns=def_cols)
		tmp = tmp.drop(columns=["shot"])
  
		summed = tmp.sum()
		new_df_cols = [*new_df_cols, *summed.index]
		new_df = [*new_df, *summed.values]
		# more to get from summed dataframe
		lh_prec = summed[[i for i in summed.index if "LH_precision" in i]]
		lh_n_prec = summed[[i for i in summed.index if "LH_N_precision" in i]]
		lh_ratio_idx = [i + "-ratio" for i in summed.index if "LH_precision" in i]
		lh_ratio = [i/j for i,j in zip(lh_prec, lh_n_prec)]
  
		lh_rec = summed[[i for i in summed.index if "LH_recall" in i]]
		lh_n_rec = summed[[i for i in summed.index if "LH_N_recall" in i]]
		lh_rec_ratio_idx = [i + "-ratio" for i in summed.index if "LH_recall" in i]
		lh_rec_ratio = [i/j for i,j in zip(lh_rec, lh_n_rec)]

		he_prec = summed[[i for i in summed.index if "HE_precision" in i]]
		he_n_prec = summed[[i for i in summed.index if "HE_N_precision" in i]]
		he_ratio_idx = [i + "-ratio" for i in summed.index if "HE_precision" in i]
		he_ratio = [i/j for i,j in zip(he_prec, he_n_prec)]

		he_rec = summed[[i for i in summed.index if "HE_recall" in i]]
		he_n_rec = summed[[i for i in summed.index if "HE_N_recall" in i]]
		he_rec_ratio_idx = [i + "-ratio" for i in summed.index if "HE_recall" in i]	
		he_rec_ratio = [i/j for i,j in zip(he_rec, he_n_rec)]
  
		new_df_cols = [*new_df_cols, *lh_ratio_idx, *lh_rec_ratio_idx, *he_ratio_idx, *he_rec_ratio_idx]
		new_df = [*new_df, *lh_ratio, *lh_rec_ratio, *he_ratio, *he_rec_ratio]
  
		df_model = pd.DataFrame(np.array(new_df).reshape(1,-1), columns=new_df_cols)
		df_model = df_model.iloc[0].to_dict()

		old_info = {} # TODO fix #df[df["models_saved_as"]==m][columns].iloc[0].to_dict()

		info = {**df_model, **old_info}
		df_new.append(pd.Series(info))
		#ii += 1

		#if ii ==3:
		#	break
		
	except Exception as e:
		errors += 1
		error_models.append([m, e])
		print(e)

df_new = pd.DataFrame(df_new)
name = "results/transitions/clf_upgrade-best_4-lab-shift_IT" # "results/clf_upgrade-sota"
if SIMPLE:
	name += "_small"
if SMOOTH != False:
    name += f"_{SMOOTH}"
if MASK != 0:
    name += f"_mask-{MASK}"
name += ".csv"
df_new.to_csv(name)

print("errors",errors)

print(df_new)







