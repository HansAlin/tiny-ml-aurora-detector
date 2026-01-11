import os
import argparse
from pathlib import Path
import sys		


from tiny_ml_code.models.FC_autoencoder import ModelBuilder
from tiny_ml_code.data_handler import DictManager
from tiny_ml_code.data_set_loader import DeepDataset
from tiny_ml_code.plotting import Plotting
from tiny_ml_code.evaluating import Evaluate 

from tensorflow import keras
import time
import json
import numpy as np

class Train():
	def __init__(self, meta_data_path, data_path="./data/processed/processed_data_subset_2024-12-01--2025-11-30.pkl", ) -> None:
		self.model = None
		self.meta_data = DictManager(path=meta_data_path)
		self.mb = ModelBuilder()
		self.call_backs = []
		self.data_set = DeepDataset(data_path=data_path, 
								meta_data_path='experiments/experiment_1/meta_data.json',
								meta_data=self.meta_data
								)
		self.x_train = None 
		self.x_test = None
		self.y_train = None
		self.y_test = None

	def create_network(self, meta_data):

		model = self.mb.build_model(
				width_layer_1 = meta_data.get('width_layer_1', 64),
				width_layer_2 = meta_data.get('width_layer_2', 32),
				activation =    meta_data.get('activation', 'relu'),
				features =  len(meta_data.get('features', [])),
				latent_size =   meta_data.get('latent_size', 2),
				model_type =    meta_data.get('model_type', 'autoencoder'),
				width_layer_last=meta_data.get('width_layer_last', 10),
				output_size = meta_data.get('output_size', 1)
			)
		
		return model



	def create_dataset(self):
		self.train_ds, self.val_ds, self.test_ds = self.data_set.prepare_tf_datasets(
				supervised_learning=(self.meta_data.get('model_type') != 'autoencoder'),
				normalize=True
			)
		if self.meta_data.get('model_type') == 'classifier':
			class_counts = self.data_set.compute_class_counts(self.train_ds)
			self.class_weight = self.data_set.compute_class_weights(class_counts)

		else:
			self.class_weight = None
		
	def create_callbacks(self, patience):

		early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
		self.call_backs.append(early_stopping_cb)

		periodic_checkpoint = PeriodicCheckpoint(
				save_dir=self.meta_data.get('model_dir'),
				every_n_epochs=10,
				prefix=self.meta_data.get("model_name")
		)
		self.call_backs.append(periodic_checkpoint)

		self.call_backs.append(self.tensorboard_callback())

	def tensorboard_callback(self):
		# Root folder for logs
		logdir = os.path.join(self.meta_data.get('model_dir'), "tensorboard_logs")

		def get_run_logdir():
			# Create a timestamped subfolder
			run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
			run_dir = os.path.join(logdir, run_id)
			
			# Make sure the folder exists
			os.makedirs(run_dir, exist_ok=True)
			
			return run_dir

		run_logdir = get_run_logdir()
		print("Log directory:", run_logdir)

		tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

		return tensorboard_cb


	def create_optimizer(self):

		if self.meta_data.get('optimizer') == 'adam':
			return keras.optimizers.Adam(learning_rate = self.meta_data.get('learning_rate', 0.001))

	def train(self):

		os.makedirs(os.path.dirname(self.meta_data.get('model_dir')), exist_ok=True)

		# Check for overwriting 
		experiment_dir = Path(os.path.dirname(self.meta_data.get('model_dir')))

		if self.meta_data.get('loss_value', None) is not None or any(experiment_dir.glob("*.h5")):
			response = input("Model seems to have been trained allready! Continue and overwrite? [Y/N]: ")
			if response.lower() != 'y':
				sys.exit("Aborted to avoid overwriting files.")

		self.model = self.create_network(self.meta_data)

		if self.meta_data.get('resume_training', False) is not False:
			if self.meta_data.get("load_weights", None) is not None:
				self.model.build( input_shape=(None, len(self.meta_data.get('features'))))
				



		self.model.compile(
				optimizer=self.create_optimizer(),
				loss=self.meta_data.get('loss'),
				metrics=self.meta_data.get('metric'),
				)
		
		
		# Build the model (important for subclassed models)
		self.model.build(input_shape=(None, len(self.meta_data.get('features'))))
		
		if self.meta_data.get("load_weights", None) is not None and self.meta_data.get('resume_training', False) is False:
			pretrained_meta_data = DictManager(path=self.meta_data.get("model_load_weigths_meta_data"))
			pretrained_model = self.create_network(pretrained_meta_data)
			pretrained_model.build( input_shape=(None, len(pretrained_meta_data.get('features'))))
			pretrained_model.load_weights(self.meta_data.get("load_weights"))

			for src, tgt in zip(pretrained_model.encoder.layers,self.model.encoder.layers):
				tgt.set_weights(src.get_weights())
				tgt.trainable = False

		for batch in self.train_ds.take(1):
			if isinstance(batch, tuple):
				x_batch, y_batch = batch
				print(x_batch.shape, y_batch.shape)
			else:
				print(batch.shape)


		initial_epoch = int(self.meta_data.get('last_epoch') or 0)
		epochs = int(self.meta_data.get('epochs'))

		history = self.model.fit(
				self.train_ds,
				validation_data=self.val_ds,
				epochs=epochs,
				initial_epoch=initial_epoch,
				callbacks=self.call_backs,
				class_weight=self.class_weight
				)

		last_epoch = history.epoch[-1] + 1

		# If EncoderClassifier with pretrained weights
		if self.meta_data.get("load_weights", None) is not None and self.meta_data.get('resume_training', None) is None:
			for layer in self.model.encoder.layers:
				layer.trainable = True

			self.model.compile(
					optimizer=keras.optimizers.Adam(1e-4),  # smaller lr
					loss=self.meta_data.get('loss'),
					metrics=self.meta_data.get('metric')
				)

			history = self.model.fit(
				self.train_ds,
				validation_data=self.val_ds,
				epochs=self.meta_data.get('epochs') + last_epoch,
				initial_epoch=last_epoch,
				callbacks=self.call_backs,
				class_weight=self.class_weight
				)
			
			last_epoch = history.epoch[-1] + 1

		save_path = os.path.join( 
				self.meta_data.get("model_dir"),  
				f"{self.meta_data.get("model_name")}_final.weights.h5"
				)
		
		self.model.save_weights(save_path)

		history_path = os.path.join(self.meta_data.get('model_dir'), 'history.json')
	
		with open(history_path, 'w') as f:
			json.dump(history.history, f)

		# Test model
		print(last_epoch)
		self.meta_data['last_epoch'] = last_epoch
		loss, metric = self.model.evaluate(self.test_ds)
		self.meta_data['loss_value'] = loss
		self.meta_data['metric_value'] = metric

		# Prediction latent space
		if self.meta_data.get('model_type') == 'autoencoder':
			# Only take the inputs from the dataset
			latent_space = self.model.encoder.predict(
				self.test_ds.map(lambda x, y: x)
			)

			latent_path = os.path.join(self.meta_data.get('model_dir'), 'latent_space.npy')
			np.save(latent_path, latent_space)

		# Reconstruct test set
		reconstructed = self.model.predict(self.test_ds)

		# Gather originals
		y_list = []
		for batch in self.test_ds.as_numpy_iterator():
			if isinstance(batch, tuple):
				x_batch, y_batch = batch
				y_list.append(y_batch)
			else:
				y_list.append(batch)
		originals = np.concatenate(y_list, axis=0)

		# Ensure shapes match
		assert reconstructed.shape == originals.shape, f"{reconstructed.shape} vs {originals.shape}"


		np.save(
			os.path.join(self.meta_data.get('model_dir'), 'reconstructed_examples.npy'),
			reconstructed
		)

		np.save(
			os.path.join(self.meta_data.get('model_dir'), 'original_examples.npy'),
			originals
		)

		self.meta_data.save_dict()




class PeriodicCheckpoint(keras.callbacks.Callback):
	def __init__(self, save_dir, every_n_epochs=10, prefix="model"):
		super().__init__()
		self.save_dir = save_dir
		self.every_n_epochs = every_n_epochs
		self.prefix = prefix
		self.epoch_model_dir = save_dir + "/model_weights/"
		os.makedirs(self.epoch_model_dir, exist_ok=True)

	def on_epoch_end(self, epoch, logs=None):
		if (epoch +1) % self.every_n_epochs == 0:
			path = os.path.join(self.epoch_model_dir + f"{self.prefix}_epoch_{epoch}.weights.h5")
			self.model.save_weights(path)
			print(f"\n Saved weights to {path}")

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train autoencoder and plot results")
	parser.add_argument(
		"--data_path",
		type=str,
		default=r"data/processed/processed_data_2_2024-12-01--2025-11-30.pkl", 
		help="Path to processed dataset"
	)
	parser.add_argument(
		"--model_dir",
		type=str,
		default=r"experiments/classifier_experiment_1", 
		help="Directory to save model weights and outputs"
	)

	args = parser.parse_args()

	meta_data_path = args.model_dir + '/meta_data.json'

	os.makedirs(os.path.dirname(meta_data_path), exist_ok=True)
	os.makedirs(args.model_dir, exist_ok=True)



	train = Train(
		meta_data_path=meta_data_path,
		data_path=args.data_path
	)

	train.create_dataset()
	train.create_callbacks(patience=10)
	train.train()

	evaluator = Evaluate(y_pred=None, y_true=None, meta_data_path=meta_data_path)
	evaluator.collect_metrics(fpr_threshold=1e-5)

	plotting = Plotting(meta_data_path=meta_data_path)
	plotting.plot_results()

