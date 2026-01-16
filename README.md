TinyML Aurora Classifier
=======================

This project is configured and controlled through a meta_data.json file.
Training, resuming, and inheriting models all start from this file.


------------------------------------------------------------
## 1. Configuration File: meta_data.json
------------------------------------------------------------

Example configuration:
```
{
  "model_dir": "experiments/classifier_experiment_6",
  "load_weights": "experiments/experiment_6/Autoencoder_final.weights.h5",
  "model_load_weigths_meta_data": "experiments/experiment_6/meta_data.json",
  "resume_training": false,

  "width_layer_1": 512,
  "width_layer_2": 256,
  "width_last_layer": 256,
  "latent_size": 64,

  "model_type": "classifier",
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

  "model_name": "Encoder-Classifier",
  "optimizer": "adam",
  "loss": "mse",
  "metric": ["accuracy"],
  "learning_rate": 0.001,
  "epochs": 200,
  "batch_size": 32
}
```

------------------------------------------------------------
## 2. Directory and Checkpoint Handling
------------------------------------------------------------

### model_dir
  Path where all outputs for this experiment are stored.

  Recommendation:
  Place meta_data.json inside this directory and set model_dir
  to the same path.


### load_weights (optional)
  Path to pretrained model weights.

  - Can be omitted if training from scratch
  - Required when resuming training or initializing from
    a pretrained model


### model_load_weigths_meta_data (conditional)
  Path to the meta_data.json file of the model from which
  weights are inherited.

  - Required if load_weights is defined
  - Can be omitted otherwise


### resume_training
  Set to true if training was interrupted and should be resumed.

  Typical workflow:
    1. Edit the current meta_data.json
    2. Add:
       - load_weights
       - model_load_weigths_meta_data
       - resume_training = true


------------------------------------------------------------
## 3. Model Architecture Parameters
------------------------------------------------------------

### model_type
  Defines the model architecture.

  Supported values:
    - autoencoder
    - classifier


### width_layer_1
  Width of the first encoder layer.
  For autoencoders, this is also the second-to-last decoder layer.


### width_layer_2
  Width of the second encoder layer.
  For autoencoders, this is also the third-to-last decoder layer.


### latent_size
  Dimensionality of the latent space.
  Only used for autoencoders.


### width_last_layer
  Size of the final layer in a classifier.
  Ignored for pure autoencoder models.


### activation
  Activation function used in all hidden layers.

  Notes:
    - Latent layer uses None
    - Output layer defaults to None

  Example:
    activation = leaky_relu


------------------------------------------------------------
### 4. Input Features
------------------------------------------------------------

### features
  List of input features used by the model.
  The order of features must match the input data.


------------------------------------------------------------
## 5. Training Configuration
------------------------------------------------------------

### model_name
Human-readable name for the model.

### optimizer
Optimizer used during training (e.g. adam).

### loss
Loss function used for optimization (e.g. mse).

### metric
List of evaluation metrics (e.g. accuracy).

### learning_rate
Learning rate passed to the optimizer.

### epochs
Number of training epochs.

### batch_size
Batch size used during training.


------------------------------------------------------------
## 6. Folder structure
------------------------------------------------------------


Root directory contains the following folders:

1. data/
   Contains all dataset files.
   - raw/
     Raw, unprocessed data.
   - processed/
     Preprocessed and cleaned data.
      - processed_data_2_2024-12-01--2025-11-30.pkl

2. experiments/
   Contains experiment outputs and trained models.

3. tiny_ml_code/
   Main source code directory.

   3.1 models/
       - FC_autoencoder.py 

       Files:
       - __init__.py
       - data_handler.py
       - data_set_loader.py
       - evaluating.py
       - get_tiny_models.ipynb
       - keras_to_pico.py
       - plotting.py
       - test.py
       - train.py


------------------------------------------------------------
## 6. Train
------------------------------------------------------------
1. Makue sure you got the data
Run this bash:
```
python train.py \
  --data_path data/processed/processed_data_2_2024-12-01--2025-11-30.pkl \
  --model_dir experiments/experiment_1

```
with your meta_data.json in experiment_1 folder.
Ensure that meta_data.json is inside experiments/experiment_1 and contains all required fields for your model.

------------------------------------------------------------
## 7. Test
------------------------------------------------------------

In order to test the models reliable run the script in test.py.
Run from project root:
	# Example of usage:
	# $ python -m tiny_ml_code.test --meta_data_path "C:/Users/hansa/Kod/Tiny ML Aurora Detector/experiments/classifier_experiment_11/meta_data.json"

------------------------------------------------------------
## 8. Quantizing models
------------------------------------------------------------
Easiest way to quantize the models are done in get_tiny_models.ipynb