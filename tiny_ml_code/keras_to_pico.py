import tensorflow as tf
from tiny_ml_code.data_set_loader import DeepDataset, DictManager
from tiny_ml_code.models.FC_autoencoder import ModelBuilder
from tiny_ml_code.plotting import Plotting
from tiny_ml_code.evaluating import Evaluate
import numpy as np


class Converter():
	def __init__(self, original_meta_data_path=None, data_path=None, tiny_ml_path=None, tiny_ml_model_name=None) -> None:
		self.meta_data_original_path = original_meta_data_path
		self.meta_data_original = DictManager(path=original_meta_data_path)
		self.dataloader = DeepDataset(data_path=data_path,
									meta_data=None,
									meta_data_path=self.meta_data_original_path)
		meta_data_tiny_ml_dict = self.meta_data_original.copy_dict()
		self.meta_data_tiny_ml = DictManager(path=tiny_ml_path + "/meta_data.json", initial=meta_data_tiny_ml_dict)
		self.meta_data_tiny_ml['model_dir'] = tiny_ml_path
		self.meta_data_tiny_ml['load_weights'] = False
		self.meta_data_tiny_ml['resume_training'] = False
		self.meta_data_tiny_ml['model_type'] = 'tiny_ml_classifier'
		self.meta_data_tiny_ml['model_name'] = tiny_ml_model_name
		self.meta_data_tiny_ml.path = tiny_ml_path + "/meta_data.json"


	def get_representative_data(self, save_subset_path="data/processed/subset_unlabeled.npz", number_of_values=200):

		representative_data = self.dataloader.load_subset(saved_subset_path=save_subset_path)

		# For supervised model, representative dataset must yield input only
		for x_batch, y_batch in representative_data.batch(1).take(number_of_values):
			yield [tf.cast(x_batch, tf.float32)]  # only the input

	def get_test_data(self, data_path) -> np.ndarray:

		self.dataloader.__setattr__('data_path', data_path)

		data_set = self.dataloader.prepare_tf_datasets(
					supervised_learning=True,
					normalize=True,
					val_fraction=0.1,
					test_fraction=0.1,
					return_numpy=True
				)

		x_test, y_test = data_set["test"]

		return x_test, y_test 

	def get_interpretter(self, tflite_model_quant):
		self.interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
		self.interpreter.allocate_tensors()

		return self.interpreter


	def test_lite_model(self, data_path, interpreter=None):

		if interpreter is None:
			interpreter = self.interpreter


		X, Y = self.get_test_data(data_path=data_path)

		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		input_scale, input_zero_point = input_details[0]["quantization"]
		output_scale, output_zero_point = output_details[0]["quantization"]


		# Quantize float32 -> int8
		X_int8 = (X / input_scale + input_zero_point).astype(np.int8)

		y_pred_float_32 = []

		for i in range(len(X_int8)):
			# Set input
			interpreter.set_tensor(input_details[0]['index'], X_int8[i:i+1])
			
			# Run inference
			interpreter.invoke()
			
			# Get output and dequantize
			output_data = interpreter.get_tensor(output_details[0]['index'])
			y_pred = (output_data.astype(np.float32) - output_zero_point) * output_scale
			y_pred_float_32.append(y_pred[0])
			
		y_pred = np.array(y_pred_float_32).flatten()

		return y_pred, Y

	def get_tflite_converter(self, weights_path="experiments/experiment_2/encoder_classifier_final.weights.h5"):
		model_builder = ModelBuilder()

		model = model_builder.build_model(
				width_layer_1 = self.meta_data_original.get('width_layer_1', 64),
				width_layer_2 = self.meta_data_original.get('width_layer_2', 32),
				activation =    self.meta_data_original.get('activation', 'relu'),
				features =  len(self.meta_data_original.get('features', [])),
				latent_size =   self.meta_data_original.get('latent_size', 2),
				model_type =    self.meta_data_original.get('model_type', 'autoencoder'),
				width_layer_last=self.meta_data_original.get('width_layer_last', 10),
				output_size = self.meta_data_original.get('output_size', 1)
			)
		
		model.load_weights(weights_path)

		converter = tf.lite.TFLiteConverter.from_keras_model(model)

		return converter
	
	def get_converted_model(self, weights_path="experiments/experiment_2/encoder_classifier_final.weights.h5"):

		converter = self.get_tflite_converter(weights_path=weights_path)

		converter.representative_dataset = tf.lite.RepresentativeDataset(self.get_representative_data)

		# Optimizations
		converter.optimizations = [tf.lite.Optimize.DEFAULT]

		converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
		# Inference input/output type
		converter.inference_input_type = tf.int8
		converter.inference_output_type = tf.int8

		tflite_model_quant = converter.convert()

		return tflite_model_quant


if __name__ == "__main__":
	experiment_nr = 1
	fpr_threshold = 1e-5
	tiny_ml_model_name = "tiny_ml_classifier"
	tiny_ml_path = f"experiments/tiny_ml_classifier_experiment_{experiment_nr}"
	original_meta_data_path = f"experiments/classifier_experiment_{experiment_nr}/meta_data.json"
	weights_to_original_model = f"experiments/classifier_experiment_{experiment_nr}/encoder_classifier_final.weights.h5"
	test_data_path = "data/processed/processed_data_2_2024-12-01--2025-11-30.pkl"
	data_for_digitalization = "data/processed/labeled_data.npz"
	converter = Converter(original_meta_data_path=original_meta_data_path,
					   data_path=data_for_digitalization,
					   tiny_ml_path=tiny_ml_path,
					   tiny_ml_model_name=tiny_ml_model_name)
	
	plotting = Plotting(meta_data_path=None, meta_data=converter.meta_data_tiny_ml)
	evaluating = Evaluate(y_pred=None, y_true=None, meta_data_path=None, meta_data=converter.meta_data_tiny_ml)

	tflite_model_quant = converter.get_converted_model(weights_path=weights_to_original_model)
	converter.get_interpretter(tflite_model_quant)
	y_pred, y_true =converter.test_lite_model(data_path=test_data_path)

	evaluating.collect_metrics(y_pred=y_pred, y_true=y_true, fpr_threshold=fpr_threshold)
	converter.meta_data_tiny_ml.save_dict(path=tiny_ml_path+"/meta_data.json")
	fpr, tpr, thresholds = evaluating.get_roc(y_pred=y_pred, y_true=y_true)

	
	plotting.plot_confusion_matrix(y_pred=y_pred, y_true=y_true)
	plotting.plot_roc_curve(fpr_threshold=fpr_threshold, y_pred=y_pred, y_true=y_true)


