import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tiny_ml_code.data_handler import DictManager

class Evaluate():
	"""This class takes care of all evaluating of, forexample the output from a Deep-Learning prediction
		together with the original values.
		Methods:
		ROC-values: produce the values for obtaining a ROC-curve
	"""

	def __init__(self, y_pred : np.array, y_true : np.array, meta_data_path="experiments/experiment_2/meta_data.json", meta_data=None) -> None:
		self.y_pred = y_pred
		self.y_true = y_true

		if meta_data is not None:
			self.meta_data = meta_data
		else:
			self.meta_data = DictManager(path=meta_data_path)

		if y_pred is None:
			pred_path = os.path.join(self.meta_data.get("model_dir"), 'reconstructed_examples.npy')
			self.y_pred  = np.load(pred_path)
		if y_true is None:
			original_path = os.path.join(self.meta_data.get("model_dir"), 'original_examples.npy')
			self.y_true  = np.load(original_path)

		self.y_pred = self.y_pred.ravel()
		self.y_true = self.y_true.ravel()



	def roc(self,):
		
		print(self.y_true.shape)
		print(self.y_pred.shape)
		print(np.unique(self.y_true, return_counts=True))


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

		self.meta_data['roc_auc'] = roc_auc
		self.meta_data['tpr_at_fpr'] = tpr_at_fpr
		self.meta_data['fpr_threshold'] = fpr_threshold
		self.meta_data['cut_threshold'] = cut
		self.meta_data['real_fpr_value'] = real_fpr_value

		return roc_auc, tpr_at_fpr, fpr_threshold, cut, real_fpr_value

	def get_metrics(self, cut):

		y_pred_labels = (self.y_pred >= cut).astype(int)

		precision = precision_score(self.y_true, y_pred_labels)
		recall = recall_score(self.y_true, y_pred_labels)
		f1 = f1_score(self.y_true, y_pred_labels)
		cm = confusion_matrix(self.y_true, y_pred_labels)

		self.meta_data['precision'] = precision
		self.meta_data['recall'] = recall
		self.meta_data['f1_score'] = f1
		self.meta_data['confusion_matrix'] =  cm.tolist()

		return precision, recall, f1, cm
	
	def collect_metrics(self,fpr_threshold=1e-5):
		fpr, tpr, thresholds = self.roc()
		roc_auc, tpr_at_fpr, fpr_threshold, cut, real_fpr_value = self.get_cut(fpr, tpr, thresholds, fpr_threshold=fpr_threshold)
		precision, recall, f1, cm = self.get_metrics(cut)

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
			'confusion_matrix': cm.tolist()
		}





			
if __name__ == "__main__":
	from tiny_ml_code.plotting import Plotting
	evaluator = Evaluate(y_pred=None, y_true=None, meta_data_path="experiments/experiment_2/meta_data.json")
	metrics = evaluator.collect_metrics(fpr_threshold=1e-5)
	print(metrics)






