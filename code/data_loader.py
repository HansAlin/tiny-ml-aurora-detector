
import os
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf

class AuroraDatasetLoader:
	"""
	Class for loading, preprocessing, and analyzing Aurora data.
	Attributes:
	ROLLING_COLUMNS (list): Columns for which rolling means are calculated.
	FLOAT_COLUMNS (list): Columns to convert to float.
	INT_COLUMNS (list): Columns to convert to int.
	DROP_COLUMNS (list): Columns to drop during preprocessing.
	"""

	ROLLING_COLUMNS = ["Temperature (C)", "Humidity (%)"]
	FLOAT_COLUMNS = ["Aurora Points", "Temperature (C)", "Humidity (%)", "Sky Temperature (C)"]
	INT_COLUMNS = ["Filter 557nm", "No filter", "IR", "Day_Night", "label"]
	DROP_COLUMNS = ["entry_id", "Clear sky value"]


	def __init__(self, data_path="./data/raw/", save_path='./data/processed/', time_zone="Europe/Stockholm", time_column="created_at"):
		
		self.data_path = data_path
		os.makedirs(save_path, exist_ok=True)
		self.save_path = save_path

		self.time_zone = time_zone
		self.time_column = time_column

	# =========================
	# Public Methods
	# =========================
	def load_raw_data(self, raw_data_name="raw_data_2024-12-01--2025-11-30.csv", labeled_data_name="aurora_labels_2024-12-01--2025-11-30.pkl", preprocess_data=True) -> pd.DataFrame:
		"""Load raw data and labeled data, merge them on the time column,
		and optionally preprocess the merged data.
		Args:
			raw_data_name (str): The filename of the raw data CSV file.
			labeled_data_name (str): The filename of the labeled data pickle file.
			preprocess_data (bool): Whether to preprocess the merged data.
		Returns:
			pd.DataFrame: The merged (and optionally preprocessed) data.
		"""	
		raw_data_file = os.path.join(self.data_path, raw_data_name)
		labeled_data_file = os.path.join(self.data_path, labeled_data_name)

		# Load the raw data and make sure that the self.time_column column is in the correct datetime format with timezone
		raw_data = pd.read_csv(raw_data_file)
		raw_data[self.time_column] = self._parse_time(raw_data[self.time_column])

		labeled_data = pd.read_pickle(labeled_data_file)
		labeled_data[self.time_column] = self._parse_time(labeled_data[self.time_column])

		# Merge the labels into the raw data based on self.time_column 
		merged_data = raw_data.merge(labeled_data, on=self.time_column, how='left')

		# Make sure that NaN values are in correct Pandas format
		merged_data['label'] = merged_data['label'].fillna(pd.NA)
		if preprocess_data:
			merged_data = self.preprocess_data(merged_data)

		# sort_values by created_at to ensure chronological order
		merged_data = merged_data.sort_values(by=self.time_column).reset_index(drop=True)

		return merged_data
	
	def preprocess_data(self, data, window_size=20) -> pd.DataFrame:
		"""
		Preprocess the input data by:
		- Validating required columns
		- Dropping unused columns
		- Converting specified columns to numeric types
		- Clipping 'Aurora Points' to be >= 0
		- Adding rolling means for specified columns
		Args:
			data (pd.DataFrame): The input data to preprocess.
			window_size (int): The window size in minutes for calculating rolling means.
		Returns:
			pd.DataFrame: The preprocessed data.
		"""
		required = {
			self.time_column,
			'Temperature (C)',
			'Humidity (%)',
			'Aurora Points'
		}
		missing = required - set(data.columns)
		if missing:
			raise ValueError(f"Missing required columns: {missing}")

		# Remove not used columns and set the columns to suitable types
		data = data.drop(columns=self.DROP_COLUMNS)

		# Convert specified columns to numeric types, coercing errors to NaN
		for col in self.FLOAT_COLUMNS:
			data[col] = pd.to_numeric(data[col], errors='coerce')

		# Convert specified columns to integer types, coercing errors to NaN
		for col in self.INT_COLUMNS:
			data[col] = pd.to_numeric(data[col], errors='coerce', downcast='integer')
		
		# Set Aurora points to be larger than or equal to zero
		data['Aurora Points'] = data['Aurora Points'].clip(lower=0)

		# Make a rolling average of the Temperature (C)  and Humidity (%) 
		# columns with a specified window size, averaging over previous values only
		self._validate_time_index(data)
		data = self._add_rolling_means( data, columns=self.ROLLING_COLUMNS, window=f"{window_size}min")

		return data

	def time_delta_stat(self, data, verbose=False) -> tuple:
		"""Calculate the time delta between consecutive entries in seconds
		and calculates min, max, mean, std, and median time deltas.
		Args:
			data (pd.DataFrame): The input data containing a time column given by self.time_column.
		Returns:
			tuple: A tuple containing (min_time_delta, max_time_delta, mean_time_delta, std_time_delta, median_time_delta)

		"""
		data = data.copy()
		time_delta = data[self.time_column].diff().dt.total_seconds()
		min_time_delta = time_delta.min()
		max_time_delta = time_delta.max()
		mean_time_delta = time_delta.mean()
		median_time_delta = time_delta.median()
		std_time_delta = time_delta.std()

		if verbose:
			print(f"Min time delta (seconds): {min_time_delta}")
			print(f"Max time delta (seconds): {max_time_delta}")
			print(f"Mean time delta (seconds): {mean_time_delta}")
			print(f"Std time delta (seconds): {std_time_delta}")
			print(f"Median time delta (seconds): {median_time_delta}")
		return min_time_delta, max_time_delta, mean_time_delta, std_time_delta, median_time_delta

	def save_processed_data(self, data, filename="processed_data") -> None:
		"""This function saves the processed data to pkl and csv files.

		Args:
			data (pd.DataFrame): The processed data to save.
			filename (str, optional): The filename, Defaults to "processed_data".

		"""
		pkl_file = os.path.join(self.save_path, f"{filename}.pkl")
		csv_file = os.path.join(self.save_path, f"{filename}.csv")

		data.to_pickle(pkl_file)
		data.to_csv(csv_file, index=False)

	def load_processed_data(self, filename="processed_data", filetype="pkl") -> pd.DataFrame:
		"""
		Load processed data from pickle or CSV.
		
		Args:
			filename (str): Filename or full path (without extension if relative).
			filetype (str): 'pkl' or 'csv'
		"""
		# Check if filename is a full path
		if os.path.abspath(filename):
			path = filename
			# add extension if missing
			if not path.endswith(f".{filetype}"):
				path = f"{path}.{filetype}"
		else:
			# treat as relative to save_path
			path = os.path.join(self.save_path, f"{filename}.{filetype}")

		if filetype == "pkl":
			return pd.read_pickle(path)
		elif filetype == "csv":
			return pd.read_csv(path)
		else:
			raise ValueError("filetype must be 'pkl' or 'csv'")



	# =========================
	# Private/Internal Methods
	# =========================
	def _add_rolling_means(self, data, columns, window="20min") -> pd.DataFrame:
		"""
		This function adds rolling mean columns to the input DataFrame for the specified columns.
		Args:
			data (pd.DataFrame): The input data containing a time column.
			columns (list): List of column names to calculate rolling means for.
			window (str): The window size for the rolling mean (e.g., '20min').
		Returns:
			pd.DataFrame: The DataFrame with added rolling mean columns.
		"""
		df = data.set_index(self.time_column)

		for col in columns:
			df[f"Rolling Mean {col}"] = (
				df[col].rolling(window, min_periods=1).mean()
			)

		return df.reset_index()
	
	def _parse_time(self, series) -> pd.Series:
		"""Convert a series of strings to datetime with the instance timezone."""
		return pd.to_datetime(series, utc=True).dt.tz_convert(self.time_zone)

	def _validate_time_index(self, data) -> None:
		"""Check that the specified time column exists and is of datetime dtype.
			Raises:
				KeyError: If the time column is missing.
				TypeError: If the time column is not a datetime type.
		"""
		if self.time_column not in data.columns:
			raise KeyError(f"Missing required time column: {self.time_column}")
		if not pd.api.types.is_datetime64_any_dtype(data[self.time_column]):
			raise TypeError(f"Time column {self.time_column} must be datetime dtype")

class DictManager:
	"""Simple dict wrapper with explicit load/save."""

	def __init__(self, path, initial=None):
		self.path = path
		os.makedirs(os.path.dirname(path), exist_ok=True)
		self.data = initial or {}
		if os.path.exists(path):
			self.load_dict()

	def __setattr__(self, name: str, value: Any) -> None:
		if name == 'path':
			os.makedirs(os.path.dirname(value), exist_ok=True)

		super().__setattr__(name, value)

	def load_dict(self):
		"""Load dict from JSON file."""
		with open(self.path, 'r') as f:
			self.data = json.load(f)

	def save_dict(self):
		"""Save dict to JSON file."""
		with open(self.path, 'w') as f:
			json.dump(self.data, f, indent=4)

	def __getitem__(self, key):
		return self.data[key]

	def __setitem__(self, key, value):
		self.data[key] = value

	def __contains__(self, key):
		return key in self.data

	def get(self, key, default=None):
		return self.data.get(key, default)

	def update(self, d):
		self.data.update(d)


class DeepDataset(AuroraDatasetLoader):

	def __init__(self, *args, data_path='./data/processed/processed_data_subset_2024-12-01--2025-11-30.pkl', meta_data_path='experiments/experiment_1/meta_data.json',meta_data=None, **kwargs):

		super().__init__(*args, **kwargs)

		
		self.data_name = None
		self.data = None
		self.supervised = None
		self.unsupervised = None
		self.data_path = data_path

		if meta_data is None:
			self.meta_data =  DictManager(meta_data_path)
		else:
			self.meta_data = meta_data
		
		self.features = self.meta_data.get('features')

		self.columns_not_to_normalize = [self.time_column, 'label', 'Day_Night']
		self.timestamps = pd.Series(dtype='datetime64[ns]')


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

	def prepare_tf_datasets(self, supervised=False, normalize=True ):
		data = self.load_processed_data(filename=self.data_path, filetype="pkl")
		_unsupervised, _supervised = self.split_to_unsupervised_data(data=data)
		if supervised:
			_unsupervised = None
			data = _supervised
			raise NotImplementedError("Supervised data preparation not implemented yet.")
		
		else:
			supervised = None
			data = _unsupervised

		self.meta_data['total_samples'] = len(data)
		#TODO remove this hardcoding
		if 'Day_Night' in data.columns:
			data = data[data['Day_Night'] == 1]

		# Remove unused columns
		data = data[self.features + [self.time_column]]

		# Split into train and test sets
		train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
		# Save timestamps
		self.timestamps = test_df[self.time_column]

		train_df = train_df.drop(columns=[self.time_column])
		test_df = test_df.drop(columns=[self.time_column])

		# Normalize data
		if normalize:
			train_df_norm = self.normalize_data(data=train_df)
			test_df_norm = self.normalize_data(data=test_df, use_predefined_values=True)

		# Convert to TensorFlow datasets
		if supervised:
			y_train_df = train_df_norm.pop('label')
			y_test_df = test_df_norm.pop('label')

			if self.meta_data.get('features') is None:
				x_train_df = train_df_norm.drop(columns=['label'])
				x_test_df = test_df_norm.drop(columns=['label'])
				self.meta_data['features'] = x_train_df.columns.tolist()
			else:
				x_train_df = train_df_norm[self.meta_data['features']]
				x_test_df = test_df_norm[self.meta_data['features']]

			X_train = self.df_to_tf_dataset(data=x_train_df, shuffle=True, batch_size=self.meta_data['batch_size'])
			X_test = self.df_to_tf_dataset(data=x_test_df, shuffle=False, batch_size=self.meta_data['batch_size'])
			Y_train = self.df_to_tf_dataset(data=y_train_df, shuffle=True, batch_size=self.meta_data['batch_size'])
			Y_test = self.df_to_tf_dataset(data=y_test_df, shuffle=False, batch_size=self.meta_data['batch_size'])

			return X_train, Y_train, X_test, Y_test

		else:
			if self.meta_data.get('features') is None:
				self.meta_data['features'] = train_df_norm.columns.tolist()
			else:
				train_df_norm = train_df_norm[self.meta_data['features']]
				test_df_norm = test_df_norm[self.meta_data['features']]

			X_train = self.df_to_tf_dataset(data=train_df_norm, shuffle=True, batch_size=self.meta_data['batch_size'])
			X_test = self.df_to_tf_dataset(data=test_df_norm, shuffle=False, batch_size=self.meta_data['batch_size'])
			return X_train, X_test


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



if __name__ == "__main__":

	# auror_loader = AuroraDatasetLoader()
	# raw_data = auror_loader.load_raw_data(raw_data_name="data_2024-12-01_2025-11-30.csv",
	# 									labeled_data_name="aurora_labels_2.pkl",
	# 									preprocess_data=True
	# )
	# auror_loader.save_processed_data(raw_data, filename="processed_data_2_2024-12-01--2025-11-30")


	# set_maker = DeepDataset(data_path="./data/processed/processed_data_subset_2024-12-01--2025-11-30.pkl", 
	# 											 meta_data_path='experiments/experiment_1/meta_data.json',
	# 											 unused_columns=['Aurora Points', 'Day_Night', 'label']
	# 											 )
	# X_train, X_test = set_maker.prepare_tf_datasets(supervised=False, normalize=True)
	# if not set_maker.sanity_check_tf_dataset(X_train, name="X_train", num_batches=3, features=len(set_maker.meta_data['features'])):
	# 	raise ValueError("Sanity check failed for X_train")
	# set_maker.meta_data.save_dict()

	

	data_1_path = r'data\processed\processed_data_2024-12-01--2025-11-30.pkl'
	data_2_path = r'data\processed\processed_data_2_2024-12-01--2025-11-30.pkl'

	with open(data_1_path, 'rb') as f:
		data_1 = pd.read_pickle(f)
	with open(data_2_path, 'rb') as f:
		data_2 = pd.read_pickle(f)
	
	data_1 = data_1[data_1['label'].notna()]
	data_2 = data_2[data_2['label'].notna()]

	data_1 = data_1.set_index('created_at')
	data_2 = data_2.set_index('created_at')

	data_1, data_2 = data_1.align(data_2, join='inner')


	print(data_1.head())
	print(data_2.head())

	columns_1 = set(data_1.columns)
	columns_2 = set(data_2.columns)
	print(columns_1)
	print(columns_2)
	print("Columns in data_1 but not in data_2:", columns_1 - columns_2)

	compare_cols = [col for col in data_1.columns if col in data_2.columns and col != 'created_at']

	data_1 = data_1.reset_index()
	data_2 = data_2.reset_index()

	# merge once
	merged = data_1.merge(
		data_2,
		on='created_at',
		suffixes=('_data_1', '_data_2'),
		how='inner'   # explicit is good practice
	)

	differences = {}

	for col in compare_cols:
		col_1 = f'{col}_data_1'
		col_2 = f'{col}_data_2'

		diff = merged[col_1] != merged[col_2]

		if diff.any():
			differences[col] = merged.loc[diff, [
				'created_at', col_1, col_2
			]]

	# report
	if differences:
		for col, diff_df in differences.items():
			print(f"\nDifferences found in column '{col}':")
			print(diff_df)
	else:
		print("No differences found.")




 
