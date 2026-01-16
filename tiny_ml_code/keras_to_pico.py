import os
import tensorflow as tf
from tiny_ml_code.data_set_loader import DeepDataset, DictManager
from tiny_ml_code.models.FC_autoencoder import ModelBuilder
from tiny_ml_code.plotting import Plotting
from tiny_ml_code.evaluating import Evaluate
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score


class Converter():
	"""
	Converter class for adapting a Keras deep learning model to a TinyML-compatible
	TensorFlow Lite (TFLite) model, handling data preparation, model conversion,
	quantization, and inference testing.

	Attributes:
		meta_data_original_path (str): Path to the original model metadata JSON.
		meta_data_original (DictManager): Original model metadata manager.
		meta_data_tiny_ml (DictManager): Metadata for TinyML model.
		interpreter (tf.lite.Interpreter): TensorFlow Lite interpreter for inference.

	Methods:
		get_representative_data(save_subset_path, number_of_values):
			Yields representative data for TFLite quantization.
		load_npz(data_path):
			Returns test data as NumPy arrays (X_test, y_test, X_val, y_val).
		get_interpretter(tflite_model_quant):
			Initializes and returns a TFLite Interpreter for the model.
		test_lite_model(data_path, interpreter=None):
			Runs inference with a TFLite model and returns predictions and true labels.
		get_tflite_converter(weights_path):
			Builds the Keras model, loads weights, and returns a TFLiteConverter.
		get_converted_model(weights_path):
			Converts the Keras model to a quantized int8 TFLite model and returns it.

	Usage:
		converter = Converter(original_meta_data_path, data_path, tiny_ml_path, tiny_ml_model_name)
		tflite_model = converter.get_converted_model(weights_path)
		interpreter = converter.get_interpretter(tflite_model)
		y_pred, y_true = converter.test_lite_model(test_data_path)
	"""

	def __init__(self, original_meta_data_path=None, data_path=None, tiny_ml_path=None, tiny_ml_model_name=None, nr_of_test_values=1000) -> None:
		
		self.data_path = data_path
		self.nr_of_test_values = nr_of_test_values

		# Load original meta data
		self.meta_data_original_path = original_meta_data_path
		self.meta_data_original = DictManager(path=original_meta_data_path)

		# Create TinyML meta data
		meta_data_tiny_ml_dict = self.meta_data_original.copy_dict()
		self.meta_data_tiny_ml = DictManager(path=tiny_ml_path + "/meta_data.json", initial=meta_data_tiny_ml_dict, create_if_missing=True)
		self.meta_data_tiny_ml['model_dir'] = tiny_ml_path
		self.meta_data_tiny_ml['load_weights'] = False
		self.meta_data_tiny_ml['resume_training'] = False
		self.meta_data_tiny_ml['model_type'] = 'tiny_ml_classifier'
		self.meta_data_tiny_ml['model_name'] = tiny_ml_model_name
		self.meta_data_tiny_ml.path = tiny_ml_path + "/meta_data.json"
		self.meta_data_tiny_ml.save_dict()

		# Test data
		self.X_test, self.y_test, self.X_val, self.y_val = self.load_npz(data_path=data_path)

		

	def load_npz(self, data_path=r"data\processed\subset_labeled.npz"):
		""" Load test and validation data from a .npz file. """
		data = np.load(data_path, allow_pickle=True)

		self.X_test = data["X_test"]
		self.y_test = data["y_test"]

		self.X_val = data["X_val"]
		self.y_val = data["y_val"]



		return self.X_test, self.y_test, self.X_val, self.y_val


	def get_representative_data(self,):
		"""
		TFLite representative dataset generator for quantization.
		Yields one large batch of inputs covering the input distribution.
		"""


		for i_value in tf.data.Dataset.from_tensor_slices(self.X_test).batch(1).take(self.nr_of_test_values):

			i_value_f32 = tf.dtypes.cast(i_value, tf.float32)
			#print(f"Type: {i_value_f32.dtype}, Shape: {i_value_f32.shape}")
			yield [i_value_f32]



	def get_interpreter(self, tflite_model_quant):
		""" This method initializes a TensorFlow Lite Interpreter for the given quantized TFLite model.

		Args:
			tflite_model_quant (TFLite model): The quantized TFLite model.

		Returns:
			tf.lite.Interpreter: The initialized TFLite Interpreter.
		"""

		self.interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)

		return self.interpreter




	def test_lite_model(self, interpreter=None, quantized_model='int8'):
		"""Run inference on int8 TFLite model using test and validation datasets."""

		if interpreter is None:
			interpreter = self.interpreter

		if quantized_model == 'int8':

			# Run int8 inference
			test_y_pred = self.evaluate_8int_model(interpreter, self.X_test)
			val_y_pred = self.evaluate_8int_model(interpreter, self.X_val)

		elif quantized_model == 'float32' or quantized_model is None:

			# Run float32 inference
			test_y_pred = self.evaluate_32float_model(interpreter, self.X_test)
			val_y_pred = self.evaluate_32float_model(interpreter, self.X_val)

		return test_y_pred, self.y_test, val_y_pred, self.y_val

	def evaluate_8int_model(self, interpreter, x_data):

		np_x_test = np.array(x_data, dtype=np.float32)

		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		interpreter.allocate_tensors()

		# Here we manually quantize the float32 data to provide int8 inputs
		input_scale, input_zero_point = input_details[0]['quantization']
		out_scale, out_zero = output_details[0]['quantization']
		np_x_test = np_x_test / input_scale + input_zero_point

		output_data = []

		for x in np_x_test:
			interpreter.set_tensor(input_details[0]['index'], x.reshape(input_details[0]['shape']).astype("int8"))
			interpreter.invoke()
			# dequantize output
			y_q = interpreter.get_tensor(output_details[0]['index'])[0]
			y = out_scale * (y_q.astype(np.float32) - out_zero)
			output_data.append(y)

		return np.array(output_data)

	def evaluate_32float_model(self, interpreter, x_data):

		np_x_test = np.array(x_data, dtype=np.float32)

		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		interpreter.allocate_tensors()

		# Run float32 inference
		output_data = []

		for x in np_x_test:
			interpreter.set_tensor(input_details[0]['index'], x.reshape(input_details[0]['shape']))
			interpreter.invoke()
			output_data.append(interpreter.get_tensor(output_details[0]['index'])[0])

		return np.array(output_data)

	



	def get_tflite_converter(self, weights_path="experiments/experiment_2/encoder_classifier_final.weights.h5"):
		""" This method builds the Keras model, loads the weights, and returns a TFLiteConverter for the model.

		Args:
			weights_path (str, optional): Path to weights for original model. Defaults to "experiments/experiment_2/encoder_classifier_final.weights.h5".

		Returns:
			tf.lite.TFLiteConverter: The TFLiteConverter for the Keras model.
		"""
		model_builder = ModelBuilder(last_activation=None)

		model = model_builder.build_model(
				width_layer_1 = self.meta_data_original.get('width_layer_1'),
				width_layer_2 = self.meta_data_original.get('width_layer_2'),
				activation =    self.meta_data_original.get('activation'),
				features =  len(self.meta_data_original.get('features')),
				latent_size =   self.meta_data_original.get('latent_size'),
				model_type =    self.meta_data_original.get('model_type'),
				width_last_layer=self.meta_data_original.get('width_last_layer'),
				output_size = self.meta_data_original.get('output_size', 1)
			)
		
		model.load_weights(weights_path)

		converter = tf.lite.TFLiteConverter.from_keras_model(model)

		return converter
	
	def get_converted_model(self, weights_path="experiments/experiment_2/encoder_classifier_final.weights.h5", quantize='int8'):
		""" This method converts the Keras model to a quantized int8 TFLite model.

		Args:
			weights_path (str, optional): Path to weights for original model. Defaults to "experiments/experiment_2/encoder_classifier_final.weights.h5".

		Returns:
			TFLite model: The quantized int8 TFLite model.
		"""
		converter = self.get_tflite_converter(weights_path=weights_path)

		if quantize == 'int8':
			converter.representative_dataset = tf.lite.RepresentativeDataset(self.get_representative_data)

			# Optimizations
			converter.optimizations = [tf.lite.Optimize.DEFAULT]

			converter.target_spec.supported_ops =  [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
			# Inference input/output type
			converter.inference_input_type = tf.int8
			converter.inference_output_type = tf.int8

		elif quantize == 'float32':

			converter.representative_dataset = tf.lite.RepresentativeDataset(self.get_representative_data)
			converter.optimizations = [tf.lite.Optimize.DEFAULT]
		else:
			pass
		

		tflite_model_quant = converter.convert()

		return tflite_model_quant
	
	def get_metrics(self, test_y_pred, test_y_true, val_y_pred, val_y_true, fpr_threshold ):

		#### Validation ##########
		val_fpr, val_tpr, val_thresholds = roc_curve(val_y_true, val_y_pred)

		idx = np.where(val_fpr <= fpr_threshold)[0][-1]
		cut_threshold = val_thresholds[idx]

		#### Test ##################
		test_fpr, test_tpr, test_thresholds = roc_curve(test_y_true, test_y_pred)
		idx = np.argmin(np.abs(test_thresholds - cut_threshold))
		test_tpr_at_fpr = test_tpr[idx]

		test_roc_auc_value = auc(test_fpr, test_tpr)
		cm = confusion_matrix(test_y_true, (test_y_pred >= cut_threshold).astype(int), normalize='true')
		precision = precision_score(test_y_true, (test_y_pred >= cut_threshold).astype(int), zero_division=0)
		recall = recall_score(test_y_true, (test_y_pred >= cut_threshold).astype(int), zero_division=0)
		f1 = f1_score(test_y_true, (test_y_pred >= cut_threshold).astype(int), zero_division=0)
		self.meta_data_tiny_ml
		self.meta_data_tiny_ml['fpr_threshold'] = float(fpr_threshold)
		self.meta_data_tiny_ml['tpr_at_fpr'] = float(test_tpr_at_fpr)
		self.meta_data_tiny_ml['cut_threshold'] = float(cut_threshold)  if cut_threshold != np.inf else 'infinity' 
		self.meta_data_tiny_ml['precision'] = float(precision)
		self.meta_data_tiny_ml['recall'] = float(recall)
		self.meta_data_tiny_ml['f1_score'] = float(f1)
		self.meta_data_tiny_ml['roc_auc'] = float(test_roc_auc_value)
		self.meta_data_tiny_ml['confusion_matrix'] = cm.tolist()

		return {'fpr_threshold': float(fpr_threshold),
				'tpr_at_fpr': float(test_tpr_at_fpr),
				'cut_threshold': float(cut_threshold)  if cut_threshold != np.inf else 'infinity',
				'precision': float(precision),
				'recall': float(recall),
				'f1_score': float(f1),
				'roc_auc': float(test_roc_auc_value),
				'confusion_matrix': cm.tolist()
				}

	def produce_h_file(self, tflite_model):

		model_path = self.meta_data_tiny_ml.get('model_dir')
		tiny_ml_model_name = self.meta_data_tiny_ml.get('model_name')
		c_array_path = os.path.join(model_path, tiny_ml_model_name + ".h")

		# Write to a .h file as a C array
		with open(c_array_path, "w") as f:
			f.write("unsigned char model[] = {\n")
			for i, b in enumerate(tflite_model):
				f.write(f"0x{b:02x},")
				if (i + 1) % 12 == 0:
					f.write("\n")
			f.write("\n};\n")
			f.write(f"unsigned int model_len = {len(tflite_model)};\n")

		print(f"Written {c_array_path}, length = {len(tflite_model)} bytes")






if __name__ == "__main__":
	# see get_tinyml_model.py for example usage
	pass
