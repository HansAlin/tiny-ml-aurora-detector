import matplotlib.pyplot as plt
import os
import json
import numpy as np

from tiny_ml_code.data_handler import DictManager

class Plotting():
	def __init__(self, meta_data_path, ) -> None:

		self.meta_data = DictManager(path=meta_data_path)
		self.path = self.meta_data.get('model_dir', '')

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

		for key in self.history.keys():

			ax.plot(self.history[key], label=key.replace('_', ' '))
		ax.set_title('Training and validation values')	
		ax.set_xlabel('Epochs')
		ax.grid()
		ax.legend()

		save_dir = os.path.join(self.path, "plots")
		os.makedirs(save_dir, exist_ok=True)

		save_plot_path = os.path.join(save_dir, "training_history.png")
		fig.savefig(save_plot_path)
		plt.show()


	def plot_examples(self, font_size=11, figsize=(16,8)):

		plt.rcParams.update({
			"font.size": font_size,
			"axes.titlesize": font_size,
			"axes.labelsize": font_size,
			"legend.fontsize": font_size,
			"xtick.labelsize": font_size,
			"ytick.labelsize": font_size,
		})

		recon_path = os.path.join(self.path, 'reconstructed_examples_10.npy')
		recon_data = np.load(recon_path)

		original_path = os.path.join(self.path, 'original_examples_10.npy')
		original_data = np.load(original_path)


		features = self.meta_data.get('features')
		residuals = recon_data - original_data

		fig, ax = plt.subplots(1,1, figsize=figsize)

		for i, residual in enumerate(residuals):
			ax.step(range(len(residual)), residual, where='mid', label=f'Example {i+1}')

		ax.set_xticks(range(len(features)))
		ax.set_xticklabels(features, rotation=45, ha='right')  # rotate if too long
		plt.tight_layout()
		ax.grid()
		ax.legend()


		save_dir = os.path.join(self.path, "plots")
		os.makedirs(save_dir, exist_ok=True)

		save_plot_path = os.path.join(save_dir, "example_residulas.png")
		fig.savefig(save_plot_path)
		plt.show()




if __name__ == '__main__':
	plotting = Plotting(meta_data_path=r'experiments\experiment_1\meta_data.json')
	plotting.plot_examples()
	plotting.history_plot()

