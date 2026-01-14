import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tiny_ml_code.data_handler import DictManager



from numpy.linalg import inv




class Evaluate():
	"""This class takes care of all evaluating of, forexample the output from a Deep-Learning prediction
		together with the original values.
		Methods:
		ROC-values: produce the values for obtaining a ROC-curve
	"""

	def __init__(self, y_pred=None, y_true=None, meta_data_path=None, meta_data=None) -> None:
		
		

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

		fpr, tpr, thresholds = roc_curve(y_true, y_pred)

		return fpr, tpr, thresholds

	@staticmethod
	def get_cut(fpr, tpr, thresholds, fpr_threshold):

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
	
	def collect_metrics(self,fpr_threshold=1e-5, y_pred=None, y_true=None, model=None):
		if model is None:
			from tiny_ml_code.models.FC_autoencoder import ModelBuilder
			self.builder = ModelBuilder()
			model = self.builder.wrapper_build_model(meta_data=self.meta_data)

		if y_pred is not None:
			self.y_pred = y_pred
		if y_true is not None:
			self.y_true = y_true

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


		fpr, tpr, thresholds = self.get_roc(y_pred=self.y_pred, y_true=self.y_true)
		roc_auc, tpr_at_fpr, fpr_threshold, cut, real_fpr_value = self.get_cut(fpr, tpr, thresholds, fpr_threshold=fpr_threshold)
		precision, recall, f1, cm = self.get_metrics(cut)

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

class Predicter():

	def __init__(self, data_path=None, weights_file='Autoencoder_final.weights.h5', save_data=False, val_fraction=0.1, test_fraction=0.1) -> None:
		from tiny_ml_code.data_set_loader import DeepDataset
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
		if save_data:
			np.savez("data/processed/numpy_data.npz", x_train=self.x_train, y_train=self.y_train, x_val=self.x_val, y_val=self.y_val, x_test=self.x_test, y_test=self.y_test, unlabeled_data=self.unlabeled_data)

	def get_model(self, meta_data:DictManager, weights_file=None):

		from tiny_ml_code.models.FC_autoencoder import ModelBuilder
		if weights_file is None:
			weights_file = self.weights_file
		weights_path = os.path.join(meta_data.get('model_dir'), weights_file)
		builder = ModelBuilder()
		model = builder.wrapper_build_model(meta_data=meta_data, compiled=True, weights_path=weights_path)

		return model
	
	def autoencoder_roc(self, mse, y_true):

		evaluator = Evaluate(y_pred=mse, y_true=y_true)
		fpr, tpr, thresholds = evaluator.get_roc()
		return fpr, tpr, thresholds
	
	def infer(self, model, data_type='val'):
		if data_type == 'val':
			data = self.x_val
		elif data_type == 'test':
			data = self.x_test
		y_pred = model.predict(data)
		mse = np.mean(np.power(data - y_pred, 2), axis=1)

		md_val = self.mahalanobis(model, data)


		return mse, y_pred, md_val
	
	def mahalanobis(self, model, data):

		encoder = model.encoder

		Z_train = encoder.predict(self.unlabeled_data)
		Z_data = encoder.predict(data)

		mu = Z_train.mean(axis=0)
		cov = np.cov(Z_train, rowvar=False)

		cov_inv = inv(cov)

		md_val = self.mahalanobis_distance(Z_data, mu, cov_inv)

		return md_val

	def mahalanobis_distance(self, Z, mu, cov_inv):
		diff = Z - mu
		return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
		





			
if __name__ == "__main__":

	predicter = Predicter(data_path='./data/processed/processed_data_2_2024-12-01--2025-11-30.pkl', save_data=True)

	for experiment in range(1,7):
		experiment_path = f'experiments/experiment_{experiment}'
		meta_data_path = os.path.join(experiment_path, 'meta_data.json') 

		meta_data = DictManager(path=meta_data_path)

		model = predicter.get_model(meta_data=meta_data)
		mse, y_val_pred, md_val = predicter.infer(model=model)
		numpay_path = os.path.join(experiment_path, 'val_predictions.npz')
		np.savez(numpay_path, mse=mse, md_val=md_val, y_val_pred=y_val_pred, x_val=predicter.x_val, labels=predicter.y_val)
		
		mse, y_val_pred, md_test = predicter.infer(model=model, data_type='test')
		numpay_path = os.path.join(experiment_path, 'test_predictions.npz')
		np.savez(numpay_path, mse=mse, md_test=md_test, y_test_pred=y_val_pred, x_test=predicter.x_test, labels=predicter.y_test)





