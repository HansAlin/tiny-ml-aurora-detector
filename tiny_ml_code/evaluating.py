import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tiny_ml_code.data_handler import DictManager
from tiny_ml_code.models.FC_autoencoder import ModelBuilder
from tensorflow import keras


class Evaluate():
	"""This class takes care of all evaluating of, forexample the output from a Deep-Learning prediction
		together with the original values.
		Methods:
		ROC-values: produce the values for obtaining a ROC-curve
	"""

	def __init__(self, y_pred : np.array, y_true : np.array, meta_data_path="experiments/experiment_2/meta_data.json", meta_data=None) -> None:
		
		self.builder = ModelBuilder()

		self.y_pred = y_pred
		self.y_true = y_true

		if meta_data is not None:
			self.meta_data = meta_data
		elif meta_data_path is not None:
			self.meta_data = DictManager(path=meta_data_path)
		else:
			self.meta_data = None

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

		fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred)

		return fpr, tpr, thresholds

	def get_cut(self, fpr, tpr, thresholds, fpr_threshold):

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

		y_pred_labels = (self.y_pred >= cut).astype(int)

		precision = precision_score(self.y_true, y_pred_labels)
		recall = recall_score(self.y_true, y_pred_labels)
		f1 = f1_score(self.y_true, y_pred_labels)
		cm = confusion_matrix(self.y_true, y_pred_labels)

		return precision, recall, f1, cm
	
	def collect_metrics(self,fpr_threshold=1e-5, y_pred=None, y_true=None):
		if y_pred is not None:
			self.y_pred = y_pred
		if y_true is not None:
			self.y_true = y_true
		fpr, tpr, thresholds = self.get_roc()
		roc_auc, tpr_at_fpr, fpr_threshold, cut, real_fpr_value = self.get_cut(fpr, tpr, thresholds, fpr_threshold=fpr_threshold)
		precision, recall, f1, cm = self.get_metrics(cut)
		model = self.builder.wrapper_build_model(meta_data=self.meta_data)
		mult, add, mack = self.builder.get_MAC(model)

		if cut == np.inf:
			cut = 1.0

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




			
if __name__ == "__main__":
	from tiny_ml_code.plotting import Plotting
	evaluator = Evaluate(y_pred=None, y_true=None, meta_data_path="experiments/classifier_experiment_2/meta_data.json")
	evaluator.collect_metrics()






