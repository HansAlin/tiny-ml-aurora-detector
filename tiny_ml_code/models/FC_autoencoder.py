import tensorflow as tf
from tensorflow import keras
from tiny_ml_code.data_set_loader import DictManager


class Encoder(keras.Model):
	def __init__(self, width_layer_1=64, width_layer_2=32, activation='relu', input_size=8, output_size=2, ):
		super().__init__()

		self.input_size = input_size
		self.output_size = output_size

		self.hidden1 = keras.layers.Dense(width_layer_1, activation=activation)
		self.hidden2 = keras.layers.Dense(width_layer_2, activation=activation)

		self.latent = keras.layers.Dense(self.output_size, activation=None)

	def call(self, inputs):

		tf.debugging.assert_equal(
					tf.shape(inputs)[-1],
					self.input_size,
					message="Encoder input feature size mismatch"
				)

		hidden1 = self.hidden1(inputs)
		hidden2 = self.hidden2(hidden1)

		latent = self.latent(hidden2) 

		tf.debugging.assert_equal(
					tf.shape(latent)[-1],
					self.output_size,
					message="Encoder output feature size mismatch"
				)

		return latent
	
class Decoder(keras.Model):
	def __init__(self, width_layer_1=64, width_layer_2=32, activation='relu', input_size=2, output_size=8, ):
		super().__init__()

		self.input_size = input_size
		self.output_size = output_size

		self.hidden1 = keras.layers.Dense(width_layer_2, activation=activation)
		self.hidden2 = keras.layers.Dense(width_layer_1, activation=activation)

		self.final_layer = keras.layers.Dense(self.output_size, activation=None)


	def call(self, inputs):

		tf.debugging.assert_equal(
				tf.shape(inputs)[-1],
				self.input_size,
				message="Decoder input latent size mismatch"
			)

		hidden1 = self.hidden1(inputs)
		hidden2 = self.hidden2(hidden1)

		output = self.final_layer(hidden2) 

		
		tf.debugging.assert_equal(
			tf.shape(output)[-1],
			self.output_size,
			message="Decoder output feature size mismatch"
		)

		return output
	
class Autoencoder(keras.Model):
	def __init__(self, width_layer_1=20, width_layer_2=10, activation='relu', features=8, latent_size=2, ):
		super().__init__()

		self.input_size = features
		self.output_size = features
		self.latent_size = latent_size

		self.encoder = Encoder(width_layer_1=width_layer_1, width_layer_2=width_layer_2, activation=activation, input_size=features, output_size=latent_size,)
		self.decoder = Decoder(width_layer_1=width_layer_1, width_layer_2=width_layer_2, activation=activation, input_size=latent_size, output_size=features,)

	def call(self, inputs):

			
		tf.debugging.assert_equal(
				tf.shape(inputs)[-1],
				self.input_size,
				message="Autoencoder input feature size mismatch"
			)

		encoded = self.encoder(inputs)
		decoded = self.decoder(encoded)

		tf.debugging.assert_equal(
				tf.shape(decoded)[-1],
				self.output_size,
				message="Autoencoder output feature size mismatch"
			)

		return decoded
	
class EncoderClassifier(keras.Model):
	def __init__(self, width_layer_1=20, width_layer_2=10, width_layer_last=10, activation='relu', features=8, latent_size=2, output_size=1):
		super().__init__()

		self.input_size = features
		self.output_size = output_size
		self.latent_size = latent_size

		self.encoder = Encoder(width_layer_1=width_layer_1, width_layer_2=width_layer_2, activation=activation, input_size=features, output_size=latent_size,)
		self.classifier = Classifier(width_last_layer=width_layer_last, activation=activation, input_size=latent_size, output_size=output_size)

	def call(self, inputs):

			
		tf.debugging.assert_equal(
				tf.shape(inputs)[-1],
				self.input_size,
				message="Encoder Classifier input feature size mismatch"
			)

		encoded = self.encoder(inputs)
		decoded = self.classifier(encoded)

		tf.debugging.assert_equal(
				tf.shape(decoded)[-1],
				self.output_size,
				message="Encoder Classifier output feature size mismatch"
			)

		return decoded


class Classifier(keras.Model):
	def __init__(self, width_last_layer=8, activation='relu', input_size=2, output_size=1):
		super().__init__()

		self.input_size = input_size
		self.output_size = output_size
 
		self.last_hidden = keras.layers.Dense(width_last_layer, activation=activation)
		self.final_hidden = keras.layers.Dense(1, activation='sigmoid')

	def call(self, inputs):

		tf.debugging.assert_equal(
				tf.shape(inputs)[-1],
				self.input_size,
				message="Classifier input latent size mismatch"
			)

		last_hidden = self.last_hidden(inputs)
		final_output = self.final_hidden(last_hidden)

		tf.debugging.assert_equal(
				tf.shape(final_output)[-1],
				self.output_size,
				message="Classifier output feature size mismatch"
			)

		return final_output


class ModelBuilder():
	def __init__(self) -> None:
		pass

	def wrapper_build_model(self, meta_data, compiled=False, weights_path=None):

		VALID_KEYS = ['width_layer_1', 'width_layer_2', 'activation', 'features', 'latent_size', 'model_type', 'width_layer_last', 'output_size']
		filtered_data = {}

		for key in VALID_KEYS:
			if key in meta_data.data:
				if key == 'features':
					filtered_data[key] = len(meta_data.get(key))
				else:
					filtered_data[key] = meta_data.get(key)

		model = self.build_model(**filtered_data)

		if compiled:
			model.compile(optimizer=keras.optimizers.Adam(), loss=meta_data.get('loss', 'mse'), metrics=meta_data.get('metrics', ['accuracy']))
		
		if weights_path is not None:
			model.load_weights(weights_path)

		return model

	def build_model(self, width_layer_1=64, width_layer_2=32, activation='relu', features=8, latent_size=2, model_type='autoencoder', width_layer_last=10, output_size=1):

		encoder_model = Encoder(width_layer_1=width_layer_1, width_layer_2=width_layer_2, activation=activation, input_size=features, output_size=latent_size,)
		decoder_model = Decoder(width_layer_1=width_layer_1, width_layer_2=width_layer_2, activation=activation, input_size=latent_size, output_size=features,)
	
		input_size = features

		if model_type == 'encoder':
			model = encoder_model
		elif model_type == 'decoder':
			input_size = latent_size
			model = decoder_model
		elif model_type == 'autoencoder':
			model = Autoencoder(width_layer_1=width_layer_1, width_layer_2=width_layer_2, activation=activation, latent_size=latent_size, features=features, )
		elif model_type == 'classifier':
			model = EncoderClassifier(width_layer_1=width_layer_1, width_layer_2=width_layer_2, width_layer_last=width_layer_last, activation=activation, features=8, latent_size=latent_size, output_size=output_size)
		else:
			raise ValueError(f"Unknown model type: {model_type}")
		
		model(tf.zeros((1, input_size)))
		model.summary()


		return model

	def get_MAC(self, model):
		"""This method calculates the MAC of the model

		Args:
			model (keras.Model): The model needs to be compiled

		Returns:
			mult (int): The number of multiplications
			add (int): The number of additions
			mack (int): The number of MACs

		"""

		adds = []
		mults = []

		for layer in model.layers:
			if 'encoder' in layer.name or 'classifier' in layer.name or 'decoder' in layer.name:
				for sub_layer in layer.layers:
					if isinstance(sub_layer, keras.layers.Dense):
						kernel = sub_layer.kernel  # tf.Variable
						input_shape, output_shape = kernel.shape

						mult = input_shape * output_shape
						mults.append(mult)

						add = (input_shape - 1) * output_shape
						adds.append(add)

			else:
				print("Layers not considered for MAC calculation")

		total_mult = sum(mults)

		total_add = sum(adds)

		total_mack = total_mult + total_add

		return total_mult, total_add, total_mack

	
if __name__ == '__main__':
	meta_data = DictManager(path=r'experiments\classifier_experiment_2\meta_data.json')
	builder = ModelBuilder()

	model = builder.wrapper_build_model(meta_data=meta_data)
	mult, add, mack = builder.get_MAC(model)
	print(f"Mult: {mult}, Add: {add}, MAC: {mack}")

		
