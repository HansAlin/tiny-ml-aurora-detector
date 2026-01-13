import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from tiny_ml_code.plotting import Plotting
from tiny_ml_code.data_handler import DictManager

class BDTModel:
	def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42, data_path='data/processed/numpy_data.npz', meta_data_path=None) -> None:
		print("Initializing BDT Model...")
		self.model = GradientBoostingClassifier(
			n_estimators=n_estimators,
			learning_rate=learning_rate,
			max_depth=max_depth,
			subsample=subsample,
			random_state=random_state,
		)

		data = np.load(data_path)
		self.x_train = data['x_val']
		self.y_train = data['y_val']
		self.x_test = data['x_test']
		self.y_test = data['y_test']
		data = None

		self.meta_data = DictManager(meta_data_path)
		self.meta_data['model_dir'] = os.path.dirname(meta_data_path)

	def train(self, compute_weights=False):
		print("Training BDT Model...")
		if compute_weights:
			sample_weights = compute_sample_weight(class_weight='balanced', y=self.y_train)
			self.model.fit(self.x_train, self.y_train, sample_weight=sample_weights)
		else:
			self.model.fit(self.x_train, self.y_train)

	def get_cut(self, fpr_target=1e-4):
		
		self.fpr_target = fpr_target
		y_prob = self.model.predict_proba(self.x_train)[:, 1]
		fpr, tpr, thresholds = roc_curve(self.y_train, y_prob)

		# Find threshold for desired FPR
		idx = np.where(fpr <= fpr_target)[0][-1]
		cut_threshold = thresholds[idx]

		return cut_threshold
	
	def evaluate(self, cut_threshold=None):

		self.y_prob = self.model.predict_proba(self.x_test)[:, 1]
		self.y_pred = (self.y_prob >= cut_threshold).astype(int) if cut_threshold is not None else self.model.predict(self.x_test)

		cm = confusion_matrix(self.y_test, self.y_pred)
		precision = precision_score(self.y_test, self.y_pred)
		recall = recall_score(self.y_test, self.y_pred)
		f1 = f1_score(self.y_test, self.y_pred)

		fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
		roc_auc_value = auc(fpr, tpr)

		evaluation_metrics = {
			"fpr_threshold": float(self.fpr_target),
			'cut_threshold': float(cut_threshold),
			'confusion_matrix': cm.tolist(),
			'precision': float(precision),
			'recall': float(recall),
			'f1_score': float(f1),
			'roc_auc': float(roc_auc_value),
			'roc_curve': (fpr, tpr, roc_auc_value)
		}

		return evaluation_metrics
	
	def transfer_to_meta_data(self, evaluation_metrics):
		
		for key, value in evaluation_metrics.items():
			if key == 'roc_curve':
				continue
			self.meta_data[key] = value
		
		self.meta_data.save_dict()

if __name__ == "__main__":

	bdt_model = BDTModel(data_path='data/processed/numpy_data.npz', meta_data_path='experiments/bdt_1/meta_data.json')
	plotting = Plotting(meta_data=bdt_model.meta_data, meta_data_path=None)
	bdt_model.train(compute_weights=False)
	cut_threshold = bdt_model.get_cut(fpr_target=1e-4)
	evaluation_metrics = bdt_model.evaluate(cut_threshold=cut_threshold)
	bdt_model.transfer_to_meta_data(evaluation_metrics)
	plotting.plot_roc_curve(y_pred=bdt_model.y_prob, y_true=bdt_model.y_test, fpr_threshold=bdt_model.fpr_target, x_scale='linear', )
	plotting.plot_confusion_matrix(normalize=True, y_pred=bdt_model.y_pred, y_true=bdt_model.y_test, threshold=cut_threshold)
