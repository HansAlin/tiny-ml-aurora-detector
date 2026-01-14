
import os
from typing import Any
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from pathlib import Path


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

	def __init__(self, path=None, initial=None, create_if_missing=False):
		"""
		Args:
			path (str): Path to the JSON file.
			initial (dict): Initial dictionary if file doesn't exist.
			create_if_missing (bool): If True, create file with `initial` if it doesn't exist.
		"""
		self.path = path
		self.data = initial or {}

		if self.path is not None:
			# Ensure parent directory exists
			os.makedirs(os.path.dirname(self.path), exist_ok=True)

			if os.path.exists(self.path):
				# File exists → load it
				self.load_dict()
			else:
				if create_if_missing:
					# Create a new file with the initial data
					self.save_dict()
					print(f"File {self.path} did not exist. Created with initial data.")
				else:
					# File missing → raise an error
					raise FileNotFoundError(f"File {self.path} does not exist.")

	def load_dict(self):
		"""Load data from JSON file."""
		try:
			with open(self.path, 'r') as f:
				self.data = json.load(f)
		except Exception as e:
			raise RuntimeError(f"Error loading dict from {self.path}: {e}")

	def save_dict(self):
		"""Save current data to JSON file."""
		try:
			with open(self.path, 'w') as f:
				json.dump(self.data, f, indent=4)
		except Exception as e:
			raise RuntimeError(f"Error saving dict to {self.path}: {e}")


	def copy_dict(self, deep=False):
		"""Return a copy of the internal dict."""
		if deep:
			return copy.deepcopy(self.data)
		return self.data.copy()



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

	def meta_data_collector(self, search_folder, search_pattern, wanted_keys):

		rows = []

		for path in Path(search_folder).rglob(search_pattern):
			
			try:
				with open(path, 'r') as f:
					data = json.load(f)

				row = {key: data.get(key, None) for key in wanted_keys}
				rows.append(row)
			except Exception as e:
				print(f"Error reading {path}: {e}")

		return pd.DataFrame(rows)




if __name__ == "__main__":

	dict_manager = DictManager(path=None)
	df = dict_manager.meta_data_collector(
			search_folder="./experiments/",
			search_pattern="meta_data.json",
			wanted_keys=["model_dir", 
				"width_layer_1", 
				"width_layer_2", 
				"latent_size",
				"width_last_layer",
				"activation",
				"metric_value",
				"roc_auc"
				]
		)
	print(df)


 
