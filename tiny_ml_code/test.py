import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from tiny_ml_code.data_handler import DictManager
from tiny_ml_code.plotting import Plotting
from tiny_ml_code.evaluating import Predicter
import argparse




def testing(meta_data, fpr_threshold=1e-4, predicter=None):

	experiment_path = meta_data.get('model_dir')
	# Initialize Plotting object (plots will not be displayed)
	plotting = Plotting(
		meta_data_path=None,
		show_plots=False,
		meta_data=meta_data,
		override_reshaping=True
	)

	# Load model with weights
	model = predicter.get_model(meta_data=meta_data, weights_file='model_final.weights.h5')

	# Predict on validation set
	val_y_pred = model.predict(predicter.x_val)

	# Count total model parameters
	total_params = model.count_params()
	meta_data['total_parameters'] = int(total_params)

	# Validation and test labels
	val_y_true = predicter.y_val
	test_y_pred = np.load(os.path.join(experiment_path, 'reconstructed_examples.npy'))
	test_y_true = np.load(os.path.join(experiment_path, 'original_examples.npy'))

	#### Validation evaluation ####
	val_fpr, val_tpr, val_thresholds = roc_curve(val_y_true, val_y_pred)
	# Find threshold corresponding to the desired FPR
	idx = np.where(val_fpr <= fpr_threshold)[0][-1]
	cut_threshold = val_thresholds[idx]

	#### Test evaluation ####
	test_fpr, test_tpr, test_thresholds = roc_curve(test_y_true, test_y_pred)
	# Find TPR on test set at validation-derived cut threshold
	idx = np.argmin(np.abs(test_thresholds - cut_threshold))
	test_tpr_at_fpr = test_tpr[idx]

	# Compute ROC AUC for test set
	test_roc_auc_value = auc(test_fpr, test_tpr)

	# Compute normalized confusion matrix using the cut threshold
	cm = confusion_matrix(test_y_true, (test_y_pred >= cut_threshold).astype(int), normalize='true')

	# Compute precision, recall, F1 score
	precision = precision_score(test_y_true, (test_y_pred >= cut_threshold).astype(int), zero_division=0)
	recall = recall_score(test_y_true, (test_y_pred >= cut_threshold).astype(int), zero_division=0)
	f1 = f1_score(test_y_true, (test_y_pred >= cut_threshold).astype(int), zero_division=0)

	# Save evaluation metrics in metadata
	meta_data['fpr_threshold'] = float(fpr_threshold)
	meta_data['tpr_at_fpr'] = float(test_tpr_at_fpr)
	meta_data['cut_threshold'] = float(cut_threshold) if cut_threshold != np.inf else 'infinity'
	meta_data['precision'] = float(precision)
	meta_data['recall'] = float(recall)
	meta_data['f1_score'] = float(f1)
	meta_data['roc_auc'] = float(test_roc_auc_value)
	meta_data['confusion_matrix'] = cm.tolist()

	meta_data.save_dict()

	# Plot ROC curve and confusion matrix for the test set
	plotting.plot_roc_curve(
		fpr_threshold=fpr_threshold,
		y_pred=test_y_pred,
		y_true=test_y_true,
		x_scale='linear',
		prefix='test_'
	)
	plotting.plot_confusion_matrix(
		y_pred=test_y_pred,
		y_true=test_y_true,
		fpr_threshold=fpr_threshold,
		cut_threshold=cut_threshold,
		normalize=True,
		prefix='test_'
	)

if __name__ == "__main__":
	# Example of usage:
	# $ python -m tiny_ml_code.test --meta_data_path "C:\Users\hansa\Kod\Tiny ML Aurora Detector\experiments\classifier_experiment_11\meta_data.json"
	parser = argparse.ArgumentParser(description="Evaluate experiments with Predicter")
	parser.add_argument(
		"--data_path",
		type=str,
		default="./data/processed/processed_data_2_2024-12-01--2025-11-30.pkl",
		help="Path to processed dataset"
	)
	parser.add_argument(
		"--save_data",
		type=bool,
		default=True,
		help="Whether to save predictions"
	)
	parser.add_argument(
		"--meta_data_path",
		type=str,
		required=True,
		help="Path to experiment meta_data.json"
	)
	parser.add_argument(
		"--fpr_threshold",
		type=float,
		default=1e-4,
		help="False positive rate threshold for evaluation"
	)

	args = parser.parse_args()

	# Initialize Predicter
	predicter = Predicter(
		data_path=args.data_path,
		save_data=args.save_data
	)

	# Load experiment metadata
	meta_data = DictManager(path=args.meta_data_path)

	# Run testing
	testing(fpr_threshold=args.fpr_threshold, meta_data=meta_data, predicter=predicter)
