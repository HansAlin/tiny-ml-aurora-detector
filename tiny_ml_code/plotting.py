import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import json
import numpy as np

from tiny_ml_code.data_handler import DictManager

class Plotting():
	def __init__(self, meta_data_path, ) -> None:

		self.meta_data = DictManager(path=meta_data_path)
		self.path = self.meta_data.get('model_dir', '')

	def plot_results(self):
		model_type = self.meta_data.get("model_type")

		if model_type == "autoencoder":
			self.plot_examples()
			plotting.plot_latent()
		else:
			self.plot_confusion_matrix()

		self.history_plot()


	def plot_confusion_matrix(self, normalize=True, labels = ["No Aurora", "Aurora"], font_size=11, figsize=(8,5)):

		
		plt.rcParams.update({
			"font.size": font_size,
			"axes.titlesize": font_size,
			"axes.labelsize": font_size,
			"legend.fontsize": font_size,
			"xtick.labelsize": font_size,
			"ytick.labelsize": font_size,
		})

		pred_path = os.path.join(self.path, 'reconstructed_examples.npy')
		pred_data = np.load(pred_path)
		pred_data = (pred_data >= 0.5).astype(int)

		original_path = os.path.join(self.path, 'original_examples.npy')
		original_data = np.load(original_path)

		

		cm = confusion_matrix(
			original_data,
			pred_data,
			normalize="true" if normalize else None
		)

		fig, ax = plt.subplots(figsize=figsize)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
		disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d")

		ax.set_title("Confusion Matrix")

		save_dir = os.path.join(self.path, "plots")
		os.makedirs(save_dir, exist_ok=True)

		save_path = os.path.join(save_dir, "confusion_matrix.png")
		fig.savefig(save_path)
		plt.close()

	def history_plot(self, font_size=11, figsize=(8,5)):

		plt.rcParams.update({
			"font.size": font_size,
			"axes.titlesize": font_size,
			"axes.labelsize": font_size,
			"legend.fontsize": font_size,
			"xtick.labelsize": font_size,
			"ytick.labelsize": font_size,
		})

		history_path = os.path.join(self.path, 'history.json')

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

		save_plot_path = os.path.join(save_dir, "training_history.png")
		fig.savefig(save_plot_path)
		plt.close()

	def plot_examples(self, font_size=11, figsize=(16,8), nr_examples=10 ):

		plt.rcParams.update({
			"font.size": font_size,
			"axes.titlesize": font_size,
			"axes.labelsize": font_size,
			"legend.fontsize": font_size,
			"xtick.labelsize": font_size,
			"ytick.labelsize": font_size,
		})

		recon_path = os.path.join(self.path, 'reconstructed_examples.npy')
		recon_data = np.load(recon_path)

		original_path = os.path.join(self.path, 'original_examples.npy')
		original_data = np.load(original_path)


		features = self.meta_data.get('features')
		residuals = recon_data - original_data

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

		save_plot_path = os.path.join(save_dir, "example_residuals.png")
		fig.savefig(save_plot_path)
		plt.close()

	def plot_latent(self, font_size=11, figsize=(16,8)):

		plt.rcParams.update({
			"font.size": font_size,
			"axes.titlesize": font_size,
			"axes.labelsize": font_size,
			"legend.fontsize": font_size,
			"xtick.labelsize": font_size,
			"ytick.labelsize": font_size,
		})

		latent_path = os.path.join(self.path, 'latent_space.npy')
		latent_data = np.load(latent_path)[:,:2]

		fig, ax = plt.subplots(1,1, figsize=figsize)
		ax.scatter(latent_data[:,0], latent_data[:,1])
		ax.set_title('Latent space')
		plt.tight_layout()
		ax.grid()


		save_dir = os.path.join(self.path, "plots")
		os.makedirs(save_dir, exist_ok=True)

		save_plot_path = os.path.join(save_dir, "latent_space.png")
		fig.savefig(save_plot_path)
		plt.close()





if __name__ == '__main__':
	plotting = Plotting(meta_data_path=r'experiments\experiment_2\meta_data.json')
	plotting.plot_results()

