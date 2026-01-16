"""
Main Script
-----------
This script trains TinyML autoencoder or classifier models using meta_data.json
and a processed dataset. Handles training, callbacks, saving weights, 
evaluation, and plotting results.
"""

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
	"""
	Train Class
	-----------
	Handles the full training pipeline for TinyML models (autoencoder or classifier).

	Attributes:
	- model: TensorFlow/Keras model
	- meta_data: DictManager instance holding model config
	- mb: ModelBuilder instance
	- call_backs: list of Keras callbacks
	- data_set: DeepDataset instance
	- x_train, x_test, y_train, y_test: dataset placeholders
	"""

	def __init__(self, meta_data_path, data_path="./data/processed/processed_data_subset_2024-12-01--2025-11-30.pkl", ) -> None:
		"""
		Initialize the training object.

		Parameters:
		- meta_data_path: path to meta_data.json
		- data_path: path to processed dataset
		"""

		self.model = None
		self.meta_data = DictManager(path=meta_data_path)
		self.mb = ModelBuilder()
		self.call_backs = []
		self.data_set = DeepDataset(data_path=data_path, 
								meta_data_path=None,
								meta_data=self.meta_data
								)
		self.x_train = None 
		self.x_test = None
		self.y_train = None
		self.y_test = None

	def create_network(self, meta_data):
		"""
		Builds the model architecture from metadata.

		Returns:
		- model: a compiled Keras model
		"""
		model = self.mb.build_model(
				width_layer_1 = meta_data.get('width_layer_1'),
				width_layer_2 = meta_data.get('width_layer_2'),
				activation =    meta_data.get('activation'),
				features =  len(meta_data.get('features')),
				latent_size =   meta_data.get('latent_size'),
				model_type =    meta_data.get('model_type'),
				width_last_layer=meta_data.get('width_last_layer'),
				output_size = meta_data.get('output_size',1)
			)
		
		return model

	def create_dataset(self):
		"""
		Prepares TensorFlow datasets for training, validation, and testing.
		Computes class weights if it's a classifier.
		"""
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
		"""Create and attach Keras callbacks for training.

		Adds EarlyStopping, PeriodicCheckpoint, and TensorBoard callbacks
		to self.call_backs to handle stopping criteria, periodic saving,
		and logging during training.

		Args:
			patience (int): Number of epochs with no improvement after which 
							training will be stopped by EarlyStopping.

		Returns:
			None
		"""
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
		"""Create a TensorBoard callback for logging training metrics.

		Generates a timestamped subfolder under 'tensorboard_logs' in the model directory 
		to store logs for the current run.

		Args:
		None

		Returns:
		keras.callbacks.TensorBoard: Configured TensorBoard callback ready to use in model.fit().
		"""

		logdir = os.path.join(self.meta_data.get('model_dir'), "tensorboard_logs")

		def get_run_logdir():
			"""Generate a timestamped directory for the current run.

			Args:
			None

			Returns:
			str: Full path to the run-specific log directory.
			"""
			run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
			run_dir = os.path.join(logdir, run_id)
			
			os.makedirs(run_dir, exist_ok=True)
			
			return run_dir

		run_logdir = get_run_logdir()
		print("Log directory:", run_logdir)

		tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

		return tensorboard_cb

	def create_optimizer(self):
		"""Create a Keras optimizer based on metadata.

		Currently supports only Adam optimizer with learning rate defined in metadata.

		Args:
		None

		Returns:
		keras.optimizers.Optimizer: Configured Keras optimizer for model compilation.
										Returns None if the optimizer type is unsupported.
		"""
		if self.meta_data.get('optimizer') == 'adam':
			return keras.optimizers.Adam(learning_rate = self.meta_data.get('learning_rate', 0.001))

	def train(self):
		"""Train the model using the prepared datasets and metadata.

		Handles full training workflow:
		- Checks for existing weights to avoid overwriting
		- Builds model architecture
		- Loads pretrained weights if specified
		- Compiles and trains the model with callbacks
		- Saves final weights and training history
		- Evaluates model on the test set
		- Saves reconstructed outputs and latent space (for autoencoders)

		Args:
			None

		Returns:
			None
		"""
		# Ensure the experiment directory exists
		os.makedirs(os.path.dirname(self.meta_data.get('model_dir')), exist_ok=True)

		 # Check for existing weights to avoid accidental overwrite
		experiment_dir = Path(os.path.dirname(self.meta_data.get('model_dir')))

		if self.meta_data.get('loss_value', None) is not None or any(experiment_dir.glob("*.h5")):
			response = input("Model seems to have been trained allready! Continue and overwrite? [Y/N]: ")
			if response.lower() != 'y':
				sys.exit("Aborted to avoid overwriting files.")

		# Create the model architecture from metadata
		self.model = self.create_network(self.meta_data)

		# If resuming training, build model to accept input shapes
		if self.meta_data.get('resume_training', False) is not False:
			if self.meta_data.get("load_weights", None) is not None:
				self.model.build( input_shape=(None, len(self.meta_data.get('features'))))
				
		# Compile model with optimizer, loss, and metrics
		self.model.compile(
				optimizer=self.create_optimizer(),
				loss=self.meta_data.get('loss'),
				metrics=self.meta_data.get('metric'),
				)
		
		# Build the model (important for subclassed models)
		self.model.build(input_shape=(None, len(self.meta_data.get('features'))))
		
		# Load pretrained weights if specified and not resuming training
		if self.meta_data.get("load_weights", None) is not None and self.meta_data.get('resume_training', False) is False:
			pretrained_meta_data = DictManager(path=self.meta_data.get("model_load_weigths_meta_data"))
			pretrained_model = self.create_network(pretrained_meta_data)
			pretrained_model.build( input_shape=(None, len(pretrained_meta_data.get('features'))))
			pretrained_model.load_weights(self.meta_data.get("load_weights"))

			for src, tgt in zip(pretrained_model.encoder.layers,self.model.encoder.layers):
				tgt.set_weights(src.get_weights())
				tgt.trainable = False

		# Debug: print shapes of one batch
		for batch in self.train_ds.take(1):
			if isinstance(batch, tuple):
				x_batch, y_batch = batch
				print(x_batch.shape, y_batch.shape)
			else:
				print(batch.shape)

		# Set training parameters
		initial_epoch = int(self.meta_data.get('last_epoch') or 0)
		epochs = int(self.meta_data.get('epochs'))

		# Train model
		history = self.model.fit(
				self.train_ds,
				validation_data=self.val_ds,
				epochs=epochs,
				initial_epoch=initial_epoch,
				callbacks=self.call_backs,
				class_weight=self.class_weight
				)

		last_epoch = history.epoch[-1] + 1

		# If using pretrained encoder-classifier, unfreeze encoder and continue training with smaller lr
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

		# Save final model weights
		save_path = os.path.join( self.meta_data.get("model_dir"), "model_final.weights.h5")
		self.model.save_weights(save_path)

		# Save training history
		history_path = os.path.join(self.meta_data.get('model_dir'), 'history.json')
		with open(history_path, 'w') as f:
			json.dump(history.history, f)

		# Evaluate model on test dataset a more reliable test is made in test.py
		self.meta_data['last_epoch'] = last_epoch
		loss, metric = self.model.evaluate(self.test_ds)
		self.meta_data['loss_value'] = loss
		self.meta_data['metric_value'] = metric

		# If autoencoder, compute and save latent space
		if self.meta_data.get('model_type') == 'autoencoder':
			# Only take the inputs from the dataset
			latent_space = self.model.encoder.predict(
				self.test_ds.map(lambda x, y: x)
			)

			latent_path = os.path.join(self.meta_data.get('model_dir'), 'latent_space.npy')
			np.save(latent_path, latent_space)

		# Reconstruct test dataset
		reconstructed = self.model.predict(self.test_ds)

		# Gather original outputs
		y_list = []
		for batch in self.test_ds.as_numpy_iterator():
			if isinstance(batch, tuple):
				x_batch, y_batch = batch
				y_list.append(y_batch)
			else:
				y_list.append(batch)
		originals = np.concatenate(y_list, axis=0)

		# Normalize output shapes to match predictions
		def normalize_output_shape(x):
			x = np.asarray(x)

			if x.ndim == 1:
				return x

			if x.ndim == 2 and x.shape[1] == 1:
				return x[:, 0]

			return x

		originals = normalize_output_shape(originals)
		reconstructed = normalize_output_shape(reconstructed)

		 # Save reconstructed examples and originals
		np.save( os.path.join(self.meta_data.get('model_dir'), 'reconstructed_examples.npy'), reconstructed)
		np.save(os.path.join(self.meta_data.get('model_dir'), 'original_examples.npy'),	originals)

		# Save updated metadata
		self.meta_data.save_dict()

class PeriodicCheckpoint(keras.callbacks.Callback):
	"""Custom Keras callback to save model weights periodically during training.

	Saves model weights every `every_n_epochs` epochs to a specified directory.
	Useful for checkpointing long training runs without losing progress.

	Attributes:
	-----------
	save_dir : str
	Directory where model weights and checkpoints will be saved.
	every_n_epochs : int
	Frequency of epochs to save model weights.
	prefix : str
	Prefix for saved model filenames.
	epoch_model_dir : str
	Full path to the subdirectory where epoch weights are saved.
	"""

	def __init__(self, save_dir, every_n_epochs=10, prefix="model"):
		"""Initialize the periodic checkpoint callback.

		Args:
		save_dir (str): Base directory to save weights.
		every_n_epochs (int, optional): Save weights every n epochs. Defaults to 10.
		prefix (str, optional): Filename prefix for saved weights. Defaults to "model".

		Returns:
		None
		"""
		super().__init__()
		self.save_dir = save_dir
		self.every_n_epochs = every_n_epochs
		self.prefix = prefix
		self.epoch_model_dir = save_dir + "/model_weights/"
		os.makedirs(self.epoch_model_dir, exist_ok=True)

	def on_epoch_end(self, epoch, logs=None):
		"""Save model weights at the end of specified epochs.

		This method is automatically called by Keras after each epoch during training.

		Args:
		epoch (int): Index of the current epoch (0-based).
		logs (dict, optional): Metric results from the training epoch. Defaults to None.

		Returns:
		None
		"""
		if (epoch +1) % self.every_n_epochs == 0:
			path = os.path.join(self.epoch_model_dir + f"{self.prefix}_epoch_{epoch}.weights.h5")
			self.model.save_weights(path)
			print(f"\n Saved weights to {path}")

if __name__ == "__main__":
	# Parse command-line arguments
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
		default=r"experiments/classifier_experiment_11", 
		help="Directory to save model weights and outputs"
	)
	args = parser.parse_args()

	# Construct path for meta_data.json inside the model directory
	meta_data_path = os.path.join(args.model_dir, "meta_data.json")

	# Make sure directories exist
	os.makedirs(os.path.dirname(meta_data_path), exist_ok=True)
	os.makedirs(args.model_dir, exist_ok=True)

	# Initialize Train class with metadata and dataset
	train = Train(
		meta_data_path=meta_data_path,
		data_path=args.data_path
	)

	# Prepare datasets (train/validation/test)
	train.create_dataset()

	# Create Keras callbacks: EarlyStopping, Checkpoints, TensorBoard
	train.create_callbacks(patience=10)

	# Train the model
	train.train()

	# If the model is not an autoencoder, run evaluation metrics
	if train.meta_data.get('model_type') != 'autoencoder':
		evaluator = Evaluate(
			y_pred=None, 
			y_true=None, 
			meta_data=train.meta_data
		)
		evaluator.collect_metrics(fpr_threshold=1e-4)

	# Plot results: training history, ROC, confusion matrix, etc.
	plotting = Plotting(
		meta_data_path=None, 
		meta_data=train.meta_data
	)
	plotting.plot_results_collection()
