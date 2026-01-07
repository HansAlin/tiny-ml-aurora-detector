import tensorflow as tf
from tensorflow import keras

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


	def call(self, latent):

		tf.debugging.assert_equal(
				tf.shape(latent)[-1],
				self.input_size,
				message="Decoder input latent size mismatch"
			)

		hidden1 = self.hidden1(latent)
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


class ModelBuilder():
	def __init__(self) -> None:
		pass

	def build_model(self, width_layer_1=64, width_layer_2=32, activation='relu', features=8, latent_size=2, model_type='autoencoder', ):

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
		else:
			raise ValueError(f"Unknown model type: {model_type}")
		
		model(tf.zeros((1, input_size)))
		model.summary()


		return model
	
if __name__ == '__main__':
	builder = ModelBuilder()
	model = builder.build_model()
	model.save_weights(r'experiments\experiment_1\model.weights.h5')
		
