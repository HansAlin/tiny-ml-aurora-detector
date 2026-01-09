
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tiny_ml_code.data_handler import AuroraDatasetLoader, DictManager

import tensorflow as tf

class DeepDataset(AuroraDatasetLoader):


	def __init__(self, *args, data_path='./data/processed/processed_data_subset_2024-12-01--2025-11-30.pkl', meta_data_path='experiments/experiment_1/meta_data.json',meta_data=None, **kwargs):

		super().__init__(*args, **kwargs)

		
		self.data_name = None
		self.data = None
		self.supervised = None
		self.unsupervised = None
		self.data_path = data_path

		if meta_data is None and meta_data_path is not None:
			self.meta_data =  DictManager(meta_data_path)
			self.features = self.meta_data.get('features')
		elif meta_data is not None:
			self.meta_data = meta_data
			self.features = self.meta_data.get('features')
		else:
			self.meta_data = None
		


		self.columns_not_to_normalize = [self.time_column, 'label', 'Day_Night']
		self.timestamps = pd.Series(dtype='datetime64[ns]')

	def __setattr__(self, __name: str, __value: Any) -> None:
		return super().__setattr__(__name, __value)

	def load_processed_data(self, filename="processed_data", filetype="pkl"):
		self.data_name = filename
		self.meta_data['data_name'] = filename
		self.data = super().load_processed_data(filename=filename, filetype=filetype)

		return self.data

	def visulize_data(self, data):
		data.hist(bins=50, figsize=(20,15))
		plt.show()

	def normalize_data(self, data=None, use_predefined_values=False):
		if data is None:
			data = self.data

		data_copy = data.copy()
		columns_to_normalize = data_copy.columns.difference(self.columns_not_to_normalize)

		if 'normalization' not in self.meta_data:
			self.meta_data['normalization'] = {}

		for col in columns_to_normalize:
			if use_predefined_values and col in self.meta_data['normalization']:
				col_mean = self.meta_data['normalization'][col]['mean']
				col_std = self.meta_data['normalization'][col]['std']
				data_copy[col] = (data_copy[col] - col_mean) / col_std
				continue
			col_mean = data_copy[col].mean()
			col_std = data_copy[col].std()
			data_copy[col] = (data_copy[col] - col_mean) / col_std
			self.meta_data['normalization'][col] = {'mean': col_mean, 'std': col_std}
		self.meta_data.save_dict()
		self.data = data_copy

		return data_copy
	
	def denormalize_data(self, data):
		data_copy = data.copy()
		columns_to_normalize = data_copy.columns.difference(self.columns_not_to_normalize)
		for col in columns_to_normalize:
			if col in self.meta_data.get('normalization', {}):
				col_mean = self.meta_data['normalization'][col]['mean']
				col_std = self.meta_data['normalization'][col]['std']
				data_copy[col] = data_copy[col] * col_std + col_mean
			else:
				raise ValueError(f"Normalization parameters for column '{col}' not found.")
		return data_copy
	
	def split_to_unsupervised_data(self, data=None, column='label', requirement=None):
		"""
		Split dataset into unsupervised and supervised parts.

		Args:
			data (pd.DataFrame | None): Input data. Uses self.data if None.
			column (str): Label column name.
			requirement:
				- None        → unsupervised = label is NaN
				- value       → unsupervised = label == value

		Returns:
			unsupervised, supervised (pd.DataFrame, pd.DataFrame)
		"""
		if data is None:
			if self.data is None:
				raise ValueError("No data provided and self.data is None")
			data = self.data
			self.data = None

		if requirement is None:
			mask = data[column].isna()
		else:
			mask = data[column] == requirement

		unsupervised = data[mask].reset_index(drop=True)
		supervised = data[~mask].reset_index(drop=True)

		# self.unsupervised = unsupervised
		# self.supervised = supervised

		return unsupervised, supervised

	def df_to_tf_dataset(self, data, target_column=None, batch_size=32, shuffle=True):
		if target_column:
			X = data.drop(columns=[target_column]).values.astype('float32')
			y = data[target_column].values.astype('float32')
			dataset = tf.data.Dataset.from_tensor_slices((X, y))
		else:
			X = data.values.astype('float32')
			dataset = tf.data.Dataset.from_tensor_slices(X)

		if shuffle:
			dataset = dataset.shuffle(buffer_size=len(data))

		dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
		return dataset

	def prepare_tf_datasets(self, supervised_learning=False, normalize=True, val_fraction=0.1, test_fraction=0.1, return_numpy=False):

		data = self.load_processed_data(filename=self.data_path, filetype="pkl")

		unsupervised_df, supervised_df = self.split_to_unsupervised_data(data=data)

		if supervised_learning:
			data = supervised_df.copy()
		else:
			data = unsupervised_df.copy()


		self.meta_data['total_samples'] = len(data)

		# TODO remove this hardcoding
		if 'Day_Night' in data.columns:
			data = data[data['Day_Night'] == 1]

		# Keep only features + time column + label (if supervised)
		if supervised_learning:
			data = data[self.features + [self.time_column, 'label']]
		else:
			data = data[self.features + [self.time_column]]

		# Split into train/val/test
		train_val_df,  test_df = train_test_split(data, test_size=test_fraction, random_state=42, shuffle=False)
		train_df, val_df = train_test_split(train_val_df, test_size=val_fraction / (1 - test_fraction), random_state=42, shuffle=True)
		print(f"Training size: {len(train_df)/len(data)*100:.0f}%")
		print(f"Validation size: {len(val_df)/len(data)*100:.0f}%")
		print(f"Training size: {len(test_df)/len(data)*100:.0f}%")


		# Save timestamps
		self.timestamps = test_df[self.time_column]

		# Drop time column for model input
		train_df = train_df.drop(columns=[self.time_column])
		val_df  = val_df.drop(columns=[self.time_column])
		test_df  = test_df.drop(columns=[self.time_column])


		# Normalize
		if normalize:
			train_df = self.normalize_data(data=train_df)
			val_df   = self.normalize_data(val_df, use_predefined_values=True)
			test_df  = self.normalize_data(test_df, use_predefined_values=True)


		def make_ds(X, y=None, shuffle=False):
			if y is None:
				ds = tf.data.Dataset.from_tensor_slices((X, X))
			else:
				ds = tf.data.Dataset.from_tensor_slices((X, y))
			if shuffle:
				ds = ds.shuffle(len(X))
			return ds.batch(self.meta_data['batch_size']).prefetch(tf.data.AUTOTUNE)
		
		if supervised_learning:
			X_train = train_df.drop(columns=['label']).values.astype('float32')
			y_train = train_df['label'].values.astype('float32')

			X_val = val_df.drop(columns=['label']).values.astype('float32')
			y_val = val_df['label'].values.astype('float32')

			X_test = test_df.drop(columns=['label']).values.astype('float32')
			y_test = test_df['label'].values.astype('float32')

			if return_numpy:
				return {
					"train": (X_train, y_train),
					"val":   (X_val, y_val),
					"test":  (X_test, y_test),
				}

			return (
				make_ds(X_train, y_train, shuffle=True),
				make_ds(X_val, y_val),
				make_ds(X_test, y_test),
			)

		else:
			X_train = train_df.values.astype('float32')
			X_val   = val_df.values.astype('float32')
			X_test  = test_df.values.astype('float32')

			if return_numpy:
				return {
					"train": X_train,
					"val":   X_val,
					"test":  X_test,
				}
			return (
				make_ds(X_train, shuffle=True),
				make_ds(X_val),
				make_ds(X_test),
			)
		
	def save_subset(self, save_subset_path, X, y=None, n_samples=1000):

		if y is None:
			np.savez(save_subset_path, x=X[:n_samples])
		else:
			np.savez(save_subset_path, x=X[:n_samples], y=y[:n_samples])

	def load_subset(self, saved_subset_path):
		
		data = np.load(saved_subset_path)

		x = data["x"]
		y = data["y"] if "y"in data else None

		if y is None:
			ds = tf.data.Dataset.from_tensor_slices(x)
			ds = ds.map(lambda X: (X, X))
		else:
			ds = tf.data.Dataset.from_tensor_slices((x, y))
			

		return ds


	def split_train_validation(self, x_train, y_train, val_fraction=0.2, shuffle=True):

		dataset = tf.data.Dataset.zip((x_train, y_train))

		# Count samples
		total = sum(1 for _ in dataset)
		val_size = int(val_fraction * total)
		train_size = total - val_size

		if shuffle:
				dataset = dataset.shuffle(buffer_size=total, reshuffle_each_iteration=False)

		train_ds = dataset.take(train_size)
		val_ds   = dataset.skip(train_size)

		return train_ds, val_ds


	def sanity_check_tf_dataset(self, dataset, name="Dataset", num_batches=None, mean_limit=0.2, std_limit=0.2, features = 8):
		"""
		Perform a sanity check on a TensorFlow dataset.
		
		Args:
			dataset: tf.data.Dataset or tf.Tensor
			name (str): Name for printing
			num_batches (int | None): number of batches to iterate over (None = all)
		"""
		print(f"\nSanity check for {name}:")
		
		# If it's already a tensor, just wrap it
		if isinstance(dataset, tf.Tensor):
			data_np = dataset.numpy()
			total_samples = data_np.shape[0]
			print(f"Shape: {data_np.shape}")
			print(f"Number of samples: {total_samples}")
			print(f"NaNs: {np.isnan(data_np).sum()}, Infs: {np.isinf(data_np).sum()}")
			print(f"Feature-wise mean: {data_np.mean(axis=0)}")
			print(f"Feature-wise std: {data_np.std(axis=0)}")
			if np.isnan(data_np).sum() == 0 and np.isinf(data_np).sum() == 0 and np.abs(data_np.mean(axis=0)) < mean_limit and np.abs(data_np.std(axis=0) - 1) < std_limit and data_np.shape[-1] == features:
				print("Sanity check passed!")
				return True
			else:
				print("Sanity check failed!")
				return False
			
		# Accumulate data from dataset batches
		all_data = []
		for i, batch in enumerate(dataset):
			all_data.append(batch.numpy())
			if num_batches is not None and i+1 >= num_batches:
				break

		all_data = np.vstack(all_data)
		
		total_samples = all_data.shape[0]
		print(f"Shape: {all_data.shape}")
		print(f"Number of samples: {total_samples}")
		print(f"NaNs: {np.isnan(all_data).sum()}, Infs: {np.isinf(all_data).sum()}")
		print(f"Feature-wise mean: {all_data.mean(axis=0)}")
		print(f"Feature-wise std: {all_data.std(axis=0)}")
		if (
			np.isnan(all_data).sum() == 0
			and np.isinf(all_data).sum() == 0
			and np.all(np.abs(all_data.mean(axis=0)) < mean_limit)
			and np.all(np.abs(all_data.std(axis=0) - 1) < std_limit)
			and all_data.shape[-1] == features
		):
			print("Sanity check passed!")
			return True
		else:
			print("Sanity check failed!")
			return False

if __name__ == '__main__':
	dataloader = DeepDataset(
			data_path=r"data\processed\processed_data_2_2024-12-01--2025-11-30.pkl",
			meta_data_path=r"experiments\experiment_2\meta_data.json"
		)
	save_subset_path = r"data\processed\subset_labeled.npz"
	# dic = dataloader.prepare_tf_datasets( supervised_learning=False, normalize=True, val_fraction=0.1, test_fraction=0.1, return_numpy=True)
	# dataloader.save_subset(save_subset_path=save_subset_path, 
	# 					X=dic['train'][0],
	# 					y=dic['train'][0],
	# 					n_samples=500)
	representative_data = dataloader.load_subset(saved_subset_path=save_subset_path)
	for x, y in representative_data.take(1):
		print(x[:3])  # first 3 samples