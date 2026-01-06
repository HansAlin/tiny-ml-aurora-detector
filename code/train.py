import os
from code.models.FC_autoencoder import ModelBuilder
from code.data_loader import DictManager, DeepDataset
from tensorflow import keras

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
				laten_size = self.meta_data.get('laten_size'),
				model_type = self.meta_data.get('model_type'),
				name = self.meta_data.get('model_name')
			)
		
	def create_dataset(self,):
		if self.meta_data.get('model_type') == 'autoencoder':
			self.x_train, self.x_test = self.data_set.prepare_tf_datasets(supervised=False, normalize=True)
			self.y_train, self.y_test = self.x_train, self.x_test
		else:
			self.x_train, self.x_test, self.y_train, self.y_test = self.data_set.prepare_tf_datasets(supervised=True, normalize=True)

	def create_callbacks(self, patience):

		early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience, restore_best=True)
		self.call_backs.append(early_stopping_cb)

		periodic_checkpoint = PeriodicCheckpoint(
				save_dir=self.meta_data.get('model_dir'),
				every_n_epochs=10,
				prefix=self.meta_data.get("model_name")
		)
		self.call_backs.append(periodic_checkpoint)

	def train(
			self):

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

		self.model.fit(
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

