import tensorflow as tf
from tensorflow import keras

class Encoder(keras.Model):
	def __init__(self, width_layer_1=20, width_layer_2=10, activation='relu', laten_size=2, **kwargs):
		super().__init__(**kwargs)

		self.hidden1 = keras.layers.Dense(width_layer_1, activation=activation)
		self.hidden2 = keras.layers.Dense(width_layer_2, activation=activation)

		self.latenspace = keras.layers.Dense(laten_size, activation=activation)

	def call(self, inputs):

		hidden1 = self.hidden1(inputs)
		hidden2 = self.hidden2(hidden1)

		latenspace = self.latenspace(hidden2) 

		return latenspace
	
class Decoder(keras.Model):
	def __init__(self, width_layer_1=20, width_layer_2=10, activation='relu', features=8, **kwargs):
		super().__init__(**kwargs)

		self.hidden1 = keras.layers.Dense(width_layer_2, activation=activation)
		self.hidden2 = keras.layers.Dense(width_layer_1, activation=activation)

		self.final_layer = keras.layers.Dense(features, activation='sigmoid')


	def call(self, latenspace):

		hidden1 = self.hidden1(latenspace)
		hidden2 = self.hidden2(hidden1)

		output = self.final_layer(hidden2) 

		return output
	
class Autoencoder(keras.Model):
		def __init__(self, width_layer_1=20, width_layer_2=10, activation='relu', features=8, **kwargs):
			super().__init__(**kwargs)

			self.encoder = Encoder(width_layer_1=width_layer_1, width_layer_2=width_layer_2, activation=activation, **kwargs)
			self.decoder = Decoder(width_layer_1=width_layer_1, width_layer_2=width_layer_2, activation=activation, features=features, **kwargs)

		def call(self, inputs):

			encoded = self.encoder(inputs)
			decoded = self.decoder(encoded)

			return decoded


class ModelBuilder():
	def __init__(self) -> None:
		pass

	def build_model(self, width_layer_1=64, width_layer_2=32, activation='relu', features=8, laten_size=2, model_type='autoencoder', **kwargs):

		encoder_model = Encoder(width_layer_1=width_layer_1, width_layer_2=width_layer_2, activation=activation, laten_size=laten_size, **kwargs)
		decoder_model = Decoder(width_layer_1=width_layer_1, width_layer_2=width_layer_2, activation=activation, features=features, **kwargs)



		dummy_input = tf.random.normal((1, features))
	
		if model_type == 'encoder':
			model = encoder_model
		elif model_type == 'decoder':
			dummy_input = tf.random.norma((1, laten_size))
			model = decoder_model
		elif model_type == 'autoencoder':
			model = Autoencoder(width_layer_1=width_layer_1, width_layer_2=width_layer_2, activation=activation, features=features, **kwargs)
		else:
			raise ValueError(f"Unknown model type: {model_type}")
		
		model(dummy_input)
		model.summary()


		return model
	
if __name__ == '__main__':
	builder = ModelBuilder()
	model = builder.build_model(name='Autoencoder')
	model.save_weights(r'experiments\experiment_1\model.weights.h5')
		
