import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from tiny_ml_code.data_handler import DictManager
from tiny_ml_code.plotting import Plotting
from tiny_ml_code.data_set_loader import DeepDataset
from tiny_ml_code.evaluating import Predicter, Evaluate


# fpr_threshold = 1e-4
# for experiment in range(1,7):
# 	experiment_path = f'experiments/experiment_{experiment}'
# 	meta_data_path = os.path.join(experiment_path, 'meta_data.json')
# 	meta_data = DictManager(path=meta_data_path)
# 	plotting = Plotting(meta_data_path=None, show_plots=True, meta_data=meta_data)
# 	data_path = os.path.join(experiment_path, 'val_predictions.npz')
# 	data = np.load(data_path)
# 	print(f"Headers in {data_path}:", data.files)
# 	mse = data['mse']
# 	md_val = data['md_val']
# 	y_val_pred = data['y_val_pred']
# 	x_val = data['x_val']
# 	labels = data['labels']

# 	##### Validation ##########
# 	fpr_val, tpr_val, thresholds_val = roc_curve( labels, mse)

# 	# Find threshold for desired FPR
# 	idx = np.where(fpr_val <= fpr_threshold)[0][-1]
# 	cut_threshold = thresholds_val[idx]

# 	##### Test ##################
# 	test_data_path = os.path.join(experiment_path, 'test_predictions.npz')
# 	test_data = np.load(test_data_path)

# 	print(f"Headers in {test_data_path}:", test_data.files)
# 	mse_test = test_data['mse']
# 	md_test = test_data['md_test']
# 	y_test_pred = test_data['y_test_pred']
# 	x_test = test_data['x_test']
# 	test_labels = test_data['labels']

# 	test_pred = (mse_test >= cut_threshold).astype(int)

# 	cm = confusion_matrix(test_labels, test_pred, normalize='true')
# 	precision = precision_score(test_labels, test_pred, zero_division=0)
# 	recall = recall_score(test_labels, test_pred, zero_division=0)
# 	f1 = f1_score(test_labels, test_pred, zero_division=0)

# 	fpr_test, tpr_test, thresholds_test  = roc_curve(test_labels, mse_test)
# 	test_pred = (mse_test >= cut_threshold).astype(int)
# 	TP = np.sum((test_pred == 1) & (test_labels == 1))
# 	FN = np.sum((test_pred == 0) & (test_labels == 1))
# 	tpr_at_cut = TP / (TP + FN) if (TP + FN) > 0 else 0.0
# 	roc_auc_value = auc(fpr_test, tpr_test)


# 	meta_data['tpr_at_fpr'] = float(tpr_at_cut) 
# 	meta_data['fpr_threshold'] = float(fpr_threshold)
# 	meta_data['cut_threshold'] = float(cut_threshold) if cut_threshold != np.inf else 'infinity' 
# 	meta_data['precision'] = float(precision)
# 	meta_data['recall'] = float(recall)
# 	meta_data['f1_score'] = float(f1)
# 	meta_data['roc_auc'] = float(roc_auc_value)
# 	meta_data['confusion_matrix'] = cm.tolist()

# 	meta_data.save_dict()


# 	plotting.plot_roc_curve(y_pred=mse_test, y_true=test_labels, x_scale='linear', fpr_threshold=fpr_threshold)
# 	plotting.plot_confusion_matrix(y_pred=mse_test, y_true=test_labels, threshold=cut_threshold, normalize=True)


predicter = Predicter(data_path='./data/processed/processed_data_2_2024-12-01--2025-11-30.pkl', save_data=True)


fpr_threshold = 1e-4
for experiment in range(13,17):

	experiment_path = f'experiments/classifier_experiment_{experiment}'
	meta_data_path = os.path.join(experiment_path, 'meta_data.json')
	meta_data = DictManager(path=meta_data_path)
	plotting = Plotting(meta_data_path=None, show_plots=True, meta_data=meta_data)
	model = predicter.get_model(meta_data=meta_data, weights_file='model_final.weights.h5')
	val_y_pred = model.predict(predicter.x_val)
	val_y_true = predicter.y_val
	test_y_pred = np.load(os.path.join(experiment_path, 'reconstructed_examples.npy'))
	test_y_true = np.load(os.path.join(experiment_path, 'original_examples.npy'))


	#### Validation ##########
	val_fpr, val_tpr, val_thresholds = roc_curve(val_y_true, val_y_pred)

	idx = np.where(val_fpr <= fpr_threshold)[0][-1]
	cut_threshold = val_thresholds[idx]

	#### Test ##################
	test_fpr, test_tpr, test_thresholds = roc_curve(test_y_true, test_y_pred)
	idx = np.argmin(np.abs(test_thresholds - cut_threshold))
	test_tpr_at_fpr = test_tpr[idx]

	test_roc_auc_value = auc(test_fpr, test_tpr)
	cm = confusion_matrix(test_y_true, (test_y_pred >= cut_threshold).astype(int), normalize='true')
	precision = precision_score(test_y_true, (test_y_pred >= cut_threshold).astype(int), zero_division=0)
	recall = recall_score(test_y_true, (test_y_pred >= cut_threshold).astype(int), zero_division=0)
	f1 = f1_score(test_y_true, (test_y_pred >= cut_threshold).astype(int), zero_division=0)


	meta_data['fpr_threshold'] = float(fpr_threshold)
	meta_data['tpr_at_fpr'] = float(test_tpr_at_fpr)
	meta_data['cut_threshold'] = float(cut_threshold)  if cut_threshold != np.inf else 'infinity' 
	meta_data['precision'] = float(precision)
	meta_data['recall'] = float(recall)
	meta_data['f1_score'] = float(f1)
	meta_data['roc_auc'] = float(test_roc_auc_value)
	meta_data['confusion_matrix'] = cm.tolist()

	meta_data.save_dict()

	plotting.plot_roc_curve(y_pred=test_y_pred, y_true=test_y_true, x_scale='linear', fpr_threshold=fpr_threshold)
	plotting.plot_confusion_matrix(y_pred=test_y_pred, y_true=test_y_true, threshold=cut_threshold, normalize=True)




