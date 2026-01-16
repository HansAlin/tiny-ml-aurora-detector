import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from tiny_ml_code.data_handler import DictManager
from numpy.linalg import inv
import argparse


class Evaluate():
	"""
	Evaluation class for Deep Learning predictions.

	Handles:
	- Loading predictions and ground truth
	- Computing ROC curves
	- Calculating cut thresholds
	- Computing precision, recall, F1-score, confusion matrix
	- Storing metrics in meta_data
	"""

	def __init__(self, y_pred=None, y_true=None, meta_data_path=None, meta_data=None) -> None:
		"""Initialize Evaluate with predictions, ground truth, and meta_data.

		If y_pred or y_true is None, attempts to load from meta_data 'model_dir'.
		"""
		print("Initializing Evaluate class")

		self.y_pred = y_pred
		self.y_true = y_true

		if meta_data is not None:
			self.meta_data = meta_data
		elif meta_data_path is not None:
			self.meta_data = DictManager(path=meta_data_path)
		else:
			self.meta_data = None

		if self.meta_data is not None:
			print(f"Loaded meta_data from {meta_data_path}")
		else:
			print("No meta_data provided")

		# Attempt to load y_pred and y_true if not provided
		try:
			if y_pred is None and self.meta_data is not None:
				pred_path = os.path.join(self.meta_data.get("model_dir"), 'reconstructed_examples.npy')
				self.y_pred  = np.load(pred_path)
			if y_true is None and self.meta_data is not None:
				original_path = os.path.join(self.meta_data.get("model_dir"), 'original_examples.npy')
				self.y_true  = np.load(original_path)

			# Flatten arrays to 1D
			if self.y_pred is not None and self.y_true is not None:
				self.y_pred = self.y_pred.ravel()
				self.y_true = self.y_true.ravel()

			# Check if y_pred and y_true have been Loaded
			if self.y_pred is not None and self.y_true is not None:
				print(f"y_pred and y_true loaded successfully with shapes {self.y_pred.shape} and {self.y_true.shape}")
			else:
				print("y_pred or y_true not loaded.")
			
		except Exception as e:
			print(f"Could not load y_pred or y_true from meta_data path. Error: {e}")


	
	def get_roc(self, y_pred=None, y_true=None):
		"""This method computes the ROC-curve values
		It can use the y_pred and y_true provided in the class initialization

		Args:
			y_pred (np.array, optional): predicted values. Defaults to None.
			y_true (np.array, optional): true values. Defaults to None.

		Returns:
			fpr: False Positive Rate
			tpr: True Positive Rate
			thresholds: Thresholds used to compute fpr and tpr
		"""
		if y_pred is not None:
			self.y_pred = y_pred

		if y_true is not None:
			self.y_true = y_true

		fpr, tpr, thresholds = roc_curve(y_true, y_pred)

		return fpr, tpr, thresholds

	@staticmethod
	def get_cut(fpr, tpr, thresholds, fpr_threshold):
		"""Determine threshold cut corresponding to a given FPR limit.

		Returns ROC AUC, TPR at FPR threshold, chosen cut, and actual FPR.
		"""
		indices = np.where(fpr <= fpr_threshold)[-1]
		if len(indices) > 0:
			index_threshold = indices[-1]
			real_fpr_value = fpr[index_threshold]
			cut = thresholds[index_threshold]
		else:
			index_threshold = 0
			real_fpr_value = fpr[0]
			cut = thresholds[0]
			
		if cut == np.inf:
			cut = 1.0
		
		roc_auc = auc(fpr, tpr)
		tpr_at_fpr = tpr[index_threshold]

		return roc_auc, tpr_at_fpr, fpr_threshold, cut, real_fpr_value

	def get_metrics(self, cut):
		"""Compute classification metrics given a threshold cut."""

		y_pred_labels = (self.y_pred >= cut).astype(int)

		precision = precision_score(self.y_true, y_pred_labels)
		recall = recall_score(self.y_true, y_pred_labels)
		f1 = f1_score(self.y_true, y_pred_labels)
		cm = confusion_matrix(self.y_true, y_pred_labels)

		return precision, recall, f1, cm
	
	def collect_metrics(self,fpr_threshold=1e-5, y_pred=None, y_true=None, model=None):
		"""Collect and save all metrics (ROC, precision, recall, F1, confusion matrix, MACs) to meta_data."""

		# Build model if not provided
		if model is None:
			from tiny_ml_code.models.FC_autoencoder import ModelBuilder
			self.builder = ModelBuilder()
			model = self.builder.wrapper_build_model(meta_data=self.meta_data)


		# Update predictions and ground truth if provided
		if y_pred is not None:
			self.y_pred = y_pred
		if y_true is not None:
			self.y_true = y_true

		# Load predictions if missing
		try:
			if y_pred is None and self.meta_data is not None:
				pred_path = os.path.join(self.meta_data.get("model_dir"), 'reconstructed_examples.npy')
				self.y_pred  = np.load(pred_path)
			if y_true is None and self.meta_data is not None:
				original_path = os.path.join(self.meta_data.get("model_dir"), 'original_examples.npy')
				self.y_true  = np.load(original_path)

			if self.y_pred is not None and self.y_true is not None:
				self.y_pred = self.y_pred.ravel()
				self.y_true = self.y_true.ravel()
		except Exception as e:
			print(f"Could not load y_pred or y_true from meta_data path. Error: {e}")

		# Compute ROC and cut
		fpr, tpr, thresholds = self.get_roc(y_pred=self.y_pred, y_true=self.y_true)
		roc_auc, tpr_at_fpr, fpr_threshold, cut, real_fpr_value = self.get_cut(fpr, tpr, thresholds, fpr_threshold=fpr_threshold)
		precision, recall, f1, cm = self.get_metrics(cut)

		# Get MACs from model
		mult, add, mack = self.builder.get_MAC(model)

		if cut == np.inf:
			cut = 1.0

		# Save metrics to meta_data
		if self.meta_data is not None:
			self.meta_data['roc_auc'] = float(roc_auc)
			self.meta_data['tpr_at_fpr'] =  float(tpr_at_fpr)
			self.meta_data['fpr_threshold'] = float(fpr_threshold)
			self.meta_data['cut_threshold'] = float(cut)
			self.meta_data['real_fpr_value'] = float(real_fpr_value)
			self.meta_data['precision'] = float(precision)
			self.meta_data['recall'] = float(recall)
			self.meta_data['f1_score'] = float(f1)
			self.meta_data['confusion_matrix'] = cm.tolist()
			self.meta_data['multiplications'] = int(mult)
			self.meta_data['additions'] = int(add)
			self.meta_data['MACs'] = int(mack)
		self.meta_data.save_dict()

		return {
			'roc_auc': roc_auc,
			'tpr_at_fpr': tpr_at_fpr,
			'fpr_threshold': fpr_threshold,
			'cut_threshold': cut,
			'real_fpr_value': real_fpr_value,
			'precision': precision,
			'recall': recall,
			'f1_score': f1,
			'confusion_matrix': cm.tolist(),
			'multiplications': mult,
			'additions': add,
			'MACs': mack

		}

class Predicter():
	"""
	Handles prediction dataset preparation and model inference.
	"""

	def __init__(self, data_path=None, weights_file='Autoencoder_final.weights.h5', save_data=False, val_fraction=0.1, test_fraction=0.1) -> None:
		
		from tiny_ml_code.data_set_loader import DeepDataset

		# Load dataset and split into train/val/test
		data_loader = DeepDataset(data_path=data_path)
		data = data_loader.prepare_tf_datasets(
			supervised_learning=True,
			normalize=True,
			val_fraction=val_fraction,
			test_fraction=test_fraction,
			return_numpy=True
		)
		self.x_val, self.y_val = data['val']
		self.x_test, self.y_test = data['test']
		self.x_train, self.y_train = data['train']
		self.unlabeled_data = data['unlabeled']

		self.weights_file = weights_file

		# Optionally save prepared numpy arrays
		if save_data:
			np.savez("data/processed/numpy_data.npz", x_train=self.x_train, y_train=self.y_train, x_val=self.x_val, y_val=self.y_val, x_test=self.x_test, y_test=self.y_test, unlabeled_data=self.unlabeled_data)

	def get_model(self, meta_data:DictManager, weights_file=None):
		"""Build and return model from meta_data and weights."""

		from tiny_ml_code.models.FC_autoencoder import ModelBuilder
		if weights_file is None:
			weights_file = self.weights_file
		weights_path = os.path.join(meta_data.get('model_dir'), weights_file)
		builder = ModelBuilder()
		model = builder.wrapper_build_model(meta_data=meta_data, compiled=True, weights_path=weights_path)

		return model
	
	def autoencoder_roc(self, mse, y_true):
		"""Compute ROC for autoencoder MSE predictions."""

		evaluator = Evaluate(y_pred=mse, y_true=y_true)
		fpr, tpr, thresholds = evaluator.get_roc()
		return fpr, tpr, thresholds
	
	def infer(self, model, data_type='val'):
		"""Run model inference on val or test data."""

		if data_type == 'val':
			data = self.x_val
		elif data_type == 'test':
			data = self.x_test
		y_pred = model.predict(data)
		mse = np.mean(np.power(data - y_pred, 2), axis=1)

		# Mahalanobis distance evaluation
		md_val = self.mahalanobis(model, data)


		return mse, y_pred, md_val
	
	def mahalanobis(self, model, data):
		"""Compute Mahalanobis distance for given data."""
		encoder = model.encoder

		Z_train = encoder.predict(self.unlabeled_data)
		Z_data = encoder.predict(data)

		mu = Z_train.mean(axis=0)
		cov = np.cov(Z_train, rowvar=False)

		cov_inv = inv(cov)

		md_val = self.mahalanobis_distance(Z_data, mu, cov_inv)

		return md_val

	def mahalanobis_distance(self, Z, mu, cov_inv):
		"""Compute Mahalanobis distance between data Z and distribution (mu, cov_inv)."""
		diff = Z - mu
		return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
		





			
if __name__ == "__main__":
	# Example of use:
	# Note that this script is suitable for autoencoder models since it provides a test of the latent spaze.
	# For other models test.py is prefered
	# 
	# python -m tiny_ml_code.test --meta_data_path "C:\Users\hansa\Kod\Tiny ML Aurora Detector\experiments\experiment_1\meta_data.json"

	parser = argparse.ArgumentParser(description="Run inference on a single experiment using Predicter.")
	parser.add_argument("--meta_data_path", type=str, default=r"C:\Users\hansa\Kod\Tiny ML Aurora Detector\experiments\experiment_1\meta_data.json", # required=True,
						help="Path to meta_data.json for the experiment (required).")
	parser.add_argument("--data_path", type=str, default="./data/processed/processed_data_2_2024-12-01--2025-11-30.pkl",
						help="Path to processed dataset (optional).")
	parser.add_argument("--save_data",  type=bool, default=True,
						help="Whether to save the numpy arrays (optional flag).")

	args = parser.parse_args()

	# Load experiment meta_data
	meta_data = DictManager(path=args.meta_data_path)
	experiment_path = os.path.dirname(args.meta_data_path)  # automatically find experiment folder

	# Initialize Predicter
	predicter = Predicter(data_path=args.data_path, save_data=args.save_data)

	# Build model
	model = predicter.get_model(meta_data=meta_data)

	# Inference on validation set
	mse_val, y_val_pred, md_val = predicter.infer(model=model, data_type='val')
	npy_path_val = os.path.join(experiment_path, 'val_predictions.npz')
	np.savez(npy_path_val, mse=mse_val, md_val=md_val, y_val_pred=y_val_pred, x_val=predicter.x_val, labels=predicter.y_val)
	print(f"Saved validation predictions to {npy_path_val}")

	# Inference on test set
	mse_test, y_test_pred, md_test = predicter.infer(model=model, data_type='test')
	npy_path_test = os.path.join(experiment_path, 'test_predictions.npz')
	np.savez(npy_path_test, mse=mse_test, md_test=md_test, y_test_pred=y_test_pred, x_test=predicter.x_test, labels=predicter.y_test)
	print(f"Saved test predictions to {npy_path_test}")



