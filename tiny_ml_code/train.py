import os
from tiny_ml_code.models.FC_autoencoder import ModelBuilder
from tiny_ml_code.data_handler import DictManager
from tiny_ml_code.data_handler import DeepDataset
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

	def create_network(self):

		self.model = self.mb.build_model(
				width_layer_1 = self.meta_data.get('width_layer_1'),
				width_layer_2 = self.meta_data.get('width_layer_2'),
				activation = self.meta_data.get('activation'),
				features = len(self.meta_data.get('features')),
				latent_size = self.meta_data.get('latent_size'),
				model_type = self.meta_data.get('model_type'),
			)
		
	def create_dataset(self,):
		if self.meta_data.get('model_type') == 'autoencoder':
			self.x_train, self.x_test = self.data_set.prepare_tf_datasets(supervised=False, normalize=True)
			self.y_train, self.y_test = self.x_train, self.x_test
		else:
			self.x_train, self.x_test, self.y_train, self.y_test = self.data_set.prepare_tf_datasets(supervised=True, normalize=True)

	def create_callbacks(self, patience):

		early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
		self.call_backs.append(early_stopping_cb)

		periodic_checkpoint = PeriodicCheckpoint(
				save_dir=self.meta_data.get('model_dir'),
				every_n_epochs=10,
				prefix=self.meta_data.get("model_name")
		)
		self.call_backs.append(periodic_checkpoint)

		self.call_backs.append(self.tensorboard_callback)

	def tensorboard_callback(self):
		# Root folder for logs
		root_logdir = os.path.join(os.curdir, "my_logs")

		def get_run_logdir():
			# Create a timestamped subfolder
			run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
			run_dir = os.path.join(root_logdir, run_id)
			
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

		self.model.compile(
				optimizer=self.meta_data.get('optimizer'),
				loss=self.meta_data.get('loss'),
				metrics=self.meta_data.get('metric'),
				)
		
		# Build the model (important for subclassed models)
		self.model.build(input_shape=(None, len(self.meta_data.get('features'))))
		self.model.summary()
		
		self.train_ds, self.val_ds = self.data_set.split_train_validation(self.x_train, self.y_train, val_fraction=0.2)

		self.x_train = None
		self.y_train = None

		# Validatation of input_shape
		for x_batch, y_batch in self.train_ds.take(1):
			print(x_batch.shape, y_batch.shape)

		history = self.model.fit(
				self.train_ds,
				validation_data=self.val_ds,
				epochs=self.meta_data.get('epochs'),
				initial_epoch=self.meta_data.get('epoch', 0),
				callbacks=self.call_backs
				)
		save_path = os.path.join( 
				self.meta_data.get("model_dir"),  
				f"{self.meta_data.get("model_name")}_final.weights.h5"
				)
		
		self.model.save_weights(save_path)

		history_path = os.path.join(self.meta_data.get('model_dir'), 'history.json')
		with open(history_path, 'w') as f:
			json.dump(history.history, f)

		# Test model
		loss, metric = self.model.evaluate(self.x_test, self.y_test)
		self.meta_data['loss'] = loss
		self.meta_data['metric_value'] = metric

		# Prediction
		if self.meta_data.get('model_type') == 'autoencoder':
			latent_space = self.model.encoder(self.x_test).numpy()
			latent_path = os.path.join(self.meta_data.get('model_dir'), 'latent_space.npy')
			np.save(latent_path, latent_space)

		# Reconstruct test set
		reconstructed = self.model(self.x_test).numpy()
		recon_path  = os.path.join(self.meta_data.get('model_dir'), 'reconstructed_examples_10.npy')
		np.save(recon_path, reconstructed[:10])
		# Original test set
		original_test_set = self.y_test[:10]
		original_test_set_path  = os.path.join(self.meta_data.get('model_dir'), 'original_examples_10.npy')
		np.save(original_test_set_path, original_test_set)


class PeriodicCheckpoint(keras.callbacks.Callback):
	def __init__(self, save_dir, every_n_epochs=10, prefix="model"):
		super().__init()
		self.save_dir = save_dir
		self.every_n_epochs = every_n_epochs
		self.prefix = prefix
		os.makedirs(save_dir, exist_ok=True)

	def on_epoch_end(self, epoch, logs=None):
		if (epoch +1) % self.every_n_epochs == 0:
			path = os.path.join( self.save_dir, f"{self.prefix}_epoch_{epoch + 1}.weights.h5")
			self.model.save_weights(path)
			print(f"\n✅ Saved weights to {path}")

if __name__ == "__main__":
	train = Train(
		meta_data_path=r'experiments\experiment_1\meta_data.json',
		data_path=r"data\processed\processed_data_2_2024-12-01--2025-11-30.pkl")
	train.meta_data.path = r'experiments\experiment_1\meta_data.json'
	train.create_network()
	train.create_dataset()
	train.train()

