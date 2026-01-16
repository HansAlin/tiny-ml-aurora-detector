import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import os
import json
import numpy as np

from tiny_ml_code.data_handler import DictManager


class Plotting():
	def __init__(self, meta_data_path, show_plots=False, meta_data=None, y_pred=None, y_true=None, override_reshaping=False) -> None:

		self.override_reshaping = override_reshaping

		if meta_data is not None:
			self.meta_data = meta_data
		else:
			self.meta_data = DictManager(path=meta_data_path)

		
		self.path = self.meta_data.get('model_dir', '')
		self.show_plots = show_plots

		self.y_pred = None
		self.y_true = None

		try:
			if y_pred is None:
				pred_path = os.path.join(self.path, 'reconstructed_examples.npy')
				self.y_pred = np.load(pred_path)
				if self.y_pred is not None:
					print(f"Loaded predictions from {pred_path}")

			if y_true is None:
				original_path = os.path.join(self.path, 'original_examples.npy')
				self.y_true = np.load(original_path)
				if self.y_true is not None:
					print(f"Loaded true labels from {original_path}")

		except Exception as e:
			print(f"Could not load predictions or true labels during initialization: {e}")

	def update_font(self, font_size=11):
		plt.rcParams.update({
			"font.size": font_size,
			"axes.titlesize": font_size,
			"axes.labelsize": font_size,
			"legend.fontsize": font_size,
			"xtick.labelsize": font_size,
			"ytick.labelsize": font_size,
		})

	def plot_results_collection(self, fpr_threshold=1e-4):
		model_type = self.meta_data.get("model_type")

		if model_type == "autoencoder":
			self.plot_examples()
			self.plot_latent()
		else:
			self.plot_confusion_matrix(y_pred=self.y_pred, y_true=self.y_true, fpr_threshold=self.meta_data['fpr_threshold'], cut_threshold=self.meta_data['cut_threshold'],)
			self.plot_roc_curve(fpr_threshold=self.meta_data['fpr_threshold'], y_pred=self.y_pred, y_true=self.y_true, x_scale='linear')

		self.history_plot()


	def plot_confusion_matrix(self, normalize=True, labels = ["No Aurora", "Aurora"], font_size=11, figsize=(8,5), y_pred=None, y_true=None, fpr_threshold=1e-4, cut_threshold=0.5, prefix=''):

		self.update_font(font_size=font_size)

		# y_pred, y_true = self._load_predictions(None, None)

		y_pred = (y_pred >= cut_threshold).astype(int)

		cm = confusion_matrix(
			y_true,
			y_pred,
			normalize="true" if normalize else None
		)

		fig, ax = plt.subplots(figsize=figsize)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
		disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d")

		ax.set_title(rf"Confusion Matrix for $\mathrm{{FPR}} < 10^{{{int(np.log10(fpr_threshold))}}}$")

		save_dir = os.path.join(self.path, "plots")
		os.makedirs(save_dir, exist_ok=True)

		save_path = os.path.join(save_dir, f"{prefix}confusion_matrix.png")
		fig.savefig(save_path)
		if self.show_plots:
			plt.show()
		plt.close()

	def history_plot(self, font_size=11, figsize=(8,5), prefix=''):

		self.update_font(font_size=font_size)

		history_path = os.path.join(self.path, f"{prefix}history.json")

		with open(history_path, 'r') as f:
			self.history = json.load(f)

		fig, ax = plt.subplots(1,1, figsize=figsize)
		twin_ax = ax.twinx()
		twin_label = None

		for key in self.history.keys():

			label = key.replace('_', ' ')

			if 'loss' in key:
				ax.plot(self.history[key], label=label)
			else:
				twin_ax.plot(self.history[key], label=label, linestyle='--')

				if twin_label is None:
					twin_label = key.split('_')[-1].capitalize()

		ax.set_ylabel('Loss')
		twin_ax.set_ylabel(twin_label)
		ax.set_title('Training and validation values')	
		ax.set_xlabel('Epochs')
		ax.grid()
		# Get labels in same box
		lines_1, labels_1 = ax.get_legend_handles_labels()
		lines_2, labels_2 = twin_ax.get_legend_handles_labels()
		ax.legend( lines_1 + lines_2, labels_1 +labels_2, loc='best')

		save_dir = os.path.join(self.path, "plots")
		os.makedirs(save_dir, exist_ok=True)

		save_plot_path = os.path.join(save_dir, f"{prefix}training_history.png")
		fig.savefig(save_plot_path)
		if self.show_plots:
			plt.show()
		plt.close()

	def plot_examples(self, font_size=11, prefix=''):

		self.update_font(font_size=font_size)

		y_pred, y_true = self._load_predictions(None, None)


		features = self.meta_data.get('features')
		residuals = y_pred - y_true

		n_features = len(features)

		n_cols = int(np.ceil(np.sqrt(n_features))) + 1
		n_rows = int(np.ceil(np.sqrt(n_features / n_cols)))


		fig, axes = plt.subplots(n_rows, n_cols, figsize=( 3 * n_cols, 3 * n_rows), sharex=False, sharey=False)

		axes = axes.flatten()

		for i, feature in enumerate(features):
			ax = axes[i]
			ax.hist( residuals[:, i], bins=50, )
			ax.set_title(feature)
			ax.axvline(0, linestyle="--", linewidth=1)
		
		if len(axes) > n_features:
			for ax in axes[n_features:]:
				ax.remove()

		fig.suptitle('Residual Distribution per Feature')

		plt.tight_layout()

		save_dir = os.path.join(self.path, "plots")
		os.makedirs(save_dir, exist_ok=True)

		save_plot_path = os.path.join(save_dir, f"{prefix}example_residuals.png")
		fig.savefig(save_plot_path)
		if self.show_plots:
			plt.show()
		plt.close()

	def plot_latent(self, font_size=11, figsize=(16,8), prefix=''):

		self.update_font(font_size=font_size)

		latent_path = os.path.join(self.path, 'latent_space.npy')
		latent_data = np.load(latent_path)[:,:2]

		fig, ax = plt.subplots(1,1, figsize=figsize)
		ax.scatter(latent_data[:,0], latent_data[:,1])
		ax.set_title('Latent space')
		plt.tight_layout()
		ax.grid()


		save_dir = os.path.join(self.path, "plots")
		os.makedirs(save_dir, exist_ok=True)

		save_plot_path = os.path.join(save_dir, f"{prefix}latent_space.png")
		fig.savefig(save_plot_path)
		if self.show_plots:
			plt.show()
		plt.close()

	def plot_roc_curve(self, font_size=11, figsize=(8,5), fpr_threshold=1e-5, y_pred=None, y_true=None, prefix='', x_scale='log'):
		
		self.update_font(font_size=font_size)
		
		fig, ax = plt.subplots(figsize=figsize)

		#y_pred, y_true = self._load_predictions(y_pred, y_true)

		fpr, tpr, thresholds = roc_curve(y_true, y_pred)

		# Find threshold for desired FPR
		idx = np.where(fpr <= fpr_threshold)[0][-1]
		cut = thresholds[idx]
		tpr_at_fpr = tpr[idx]

		ax.plot(fpr, tpr)
		ax.plot([1e-6, 1], [1e-6, 1], linestyle='--', color='gray')

		ax.set_title(
			f"ROC-curve"
			# rf"With $\mathrm{{FPR}} < 10^{{{int(np.log10(fpr_threshold))}}}$, cut = {cut:.2f}"
			# f"\n"
			# rf"Gives TPR: {tpr_at_fpr:.2f}"
		)
		ax.set_xlabel("False Positive Rate")
		ax.set_ylabel("True Positive Rate")
		ax.set_xscale(x_scale)
		plt.grid()

		save_dir = os.path.join(self.path, "plots")
		os.makedirs(save_dir, exist_ok=True)

		save_path = os.path.join(save_dir, f"{prefix}roc_curve.png")
		plt.tight_layout()
		fig.savefig(save_path)
		if self.show_plots:
			plt.show()
		plt.close()
		
	def _load_predictions(self, y_pred=None, y_true=None):
		"""
		Lazy loader for predictions and labels.
		Explicit arguments override cached values.
		"""
		print("Loading predictions and true labels...")

		if y_pred is not None:
			self.y_pred = y_pred

		if y_true is not None:
			self.y_true = y_true

		if self.y_pred is None:
			pred_path = os.path.join(self.path, 'reconstructed_examples.npy')
			self.y_pred = np.load(pred_path)
			if self.y_pred is not None:
				print(f"Loaded predictions from {pred_path}")

		if self.y_true is None:
			true_path = os.path.join(self.path, 'original_examples.npy')
			self.y_true = np.load(true_path)
			if self.y_true is not None:
				print(f"Loaded true labels from {true_path}")

		if not self.override_reshaping:
			
			if self.meta_data.get("model_type") == "autoencoder":
				if self.y_pred.ndim == 1 and self.y_true.ndim == 1:
					# Get the features back from flattened arrays
					n_features = len(self.meta_data.get("features", []))
					self.y_pred = self.y_pred.reshape((-1, n_features))
					self.y_true = self.y_true.reshape((-1, n_features))

		return self.y_pred, self.y_true


 

if __name__ == '__main__':
	plotting = Plotting(meta_data_path=r'experiments\experiment_2\meta_data.json', show_plots=True)
	plotting.plot_results_collection()
