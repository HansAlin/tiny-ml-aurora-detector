import tensorflow as tf
from tiny_ml_code.data_set_loader import DeepDataset, DictManager
from tiny_ml_code.models.FC_autoencoder import ModelBuilder
import numpy as np

class Converter():
	def __init__(self, meta_data_path=None, data_path=None) -> None:
		self.meta_data_path = meta_data_path
		self.meta_data = DictManager(path=meta_data_path)
		self.dataloader = DeepDataset(data_path=data_path,
									meta_data=None,
									meta_data_path=self.meta_data_path)

	def get_representative_data(self, save_subset_path="data/processed/subset_unlabeled.npz", number_of_values=200):

		representative_data = self.dataloader.load_subset(saved_subset_path=save_subset_path)

		# For supervised model, representative dataset must yield input only
		for x_batch, y_batch in representative_data.batch(1).take(number_of_values):
			yield [tf.cast(x_batch, tf.float32)]  # only the input

	def get_test_data(self, data_path) -> np.ndarray:

		self.dataloader.__setattr__('data_path', data_path)

		self.dataloader.prepare_tf_datasets(
			supervised_learning=True,
			normalize=True,
			val_fraction=0.1,
			test_fraction=0.1,
			return_numpy=True
		)




	def get_tflite_converter(self, weights_path="experiments/experiment_2/encoder_classifier_final.weights.h5"):
		model_builder = ModelBuilder()

		model = model_builder.build_model(
				width_layer_1 = self.meta_data.get('width_layer_1', 64),
				width_layer_2 = self.meta_data.get('width_layer_2', 32),
				activation =    self.meta_data.get('activation', 'relu'),
				features =  len(self.meta_data.get('features', [])),
				latent_size =   self.meta_data.get('latent_size', 2),
				model_type =    self.meta_data.get('model_type', 'autoencoder'),
				width_layer_last=self.meta_data.get('width_layer_last', 10),
				output_size = self.meta_data.get('output_size', 1)
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

	converter = Converter(meta_data_path="experiments/experiment_2/meta_data.json")

	tflite_model_quant = converter.get_converted_model(weights_path="experiments/experiment_2/encoder_classifier_final.weights.h5")
	interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	scale, zero_point = input_details[0]["quantization"]

	print(scale, zero_point)
