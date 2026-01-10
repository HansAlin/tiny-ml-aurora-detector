## TinyML Aurora Classifier
Everything starts with creating a meta_data.json file
Example:
```
{
    "model_dir": "experiments/experiment_1",
    "load_weights": "experiments/experiment_1/Autoencoder_final.weights.h5",
	"model_load_weigths_meta_data": "experiments/experiment_1/meta_data.json",
    "resume_training": false,
    "model_type": "autoencoder",
    "model_name": "Autoencoder",
    "width_layer_1": 32,
    "width_layer_2": 16,
    "activation": "leaky_relu",
    "features": [
        "Filter 557nm",
        "Humidity (%)",
        "IR",
        "No filter",
        "Rolling Mean Humidity (%)",
        "Rolling Mean Temperature (C)",
        "Sky Temperature (C)",
        "Temperature (C)"
    ],
    "latent_size": 2,
    "optimizer": "adam",
    "loss": "mse",
    "metric": [
        "accuracy"
    ],
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32
}
```
### model_dir
Simplest is to use the same folder as where the meta_data.json is located
### load_weights
This is optional if one have pretrained weights can be left out otherwise
### model_load_weigths_meta_data
This is requiered if "load_weights" is defined oterwise it can be left out
### resume_training
If the traing have been interupted by some reason set this to true. Im practis one
change the current meta_data.json by adding "load_weights", "model_load_weigths_meta_data" (to this meta_data.json!) and "resume_training"


