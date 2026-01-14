import h5py
import numpy as np
import os
import json


for i in range(1,7):
	# ------------------------------------------------------------
	# PATHS
	# ------------------------------------------------------------
	BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/tiny-ml-aurora-detector"
	EXPERIMENT_PATH = os.path.join(BASE_PATH, "experiments", f"classifier_experiment_{i}")

	META_DATA_PATH = os.path.join(EXPERIMENT_PATH, "meta_data.json")
	WEIGHTS_PATH = os.path.join(EXPERIMENT_PATH, "model_final.weights.h5")

	import h5py

	WEIGHTS_PATH = f"/content/drive/MyDrive/Colab Notebooks/tiny-ml-aurora-detector/experiments/classifier_experiment_{i}/model_final.weights.h5"

	# with h5py.File(WEIGHTS_PATH, "r") as f:
	#     def walk(group, prefix=""):
	#         for k in group:
	#             item = group[k]
	#             path = f"{prefix}/{k}" if prefix else k
	#             if isinstance(item, h5py.Dataset):
	#                 print(path, item.shape)
	#             elif isinstance(item, h5py.Group):
	#                 walk(item, path)

	#     walk(f)

	# ------------------------------------------------------------
	# LOAD META DATA
	# ------------------------------------------------------------
	with open(META_DATA_PATH, "r") as f:
		meta = json.load(f)

	features = meta["features"]
	w1 = meta["width_layer_1"]
	w2 = meta["width_layer_2"]
	latent = meta["latent_size"]
	w_last = meta["width_last_layer"]
	out = meta.get("output_size", 1)
	model_type = meta["model_type"]

	# ------------------------------------------------------------
	# EXPECTED KERNEL SHAPES
	# ------------------------------------------------------------
	expected_kernels = []

	# Encoder
	expected_kernels += [
		(len(features), w1),
		(w1, w2),
		(w2, latent),
	]

	# Classifier head
	expected_kernels += [
		(latent, w_last),
		(w_last, out),
	]

	# ------------------------------------------------------------
	# HDF5 UTILITIES
	# ------------------------------------------------------------
	def collect_dense_kernels(h5file):
		kernels = {}
		def walk(group, prefix=""):
			for k in group:
				item = group[k]
				path = f"{prefix}/{k}" if prefix else k
				# Only real layer kernels, ignore optimizer vars
				if isinstance(item, h5py.Dataset) and len(item.shape) == 2 and not prefix.startswith("optimizer"):
					kernels[path] = np.array(item)
				elif isinstance(item, h5py.Group):
					walk(item, path)
		walk(h5file)
		return kernels

	# ------------------------------------------------------------
	# MAIN
	# ------------------------------------------------------------
	with h5py.File(WEIGHTS_PATH, "r") as f:
		kernels = collect_dense_kernels(f)

	found_shapes = [v.shape for v in kernels.values()]
	print(len(found_shapes))
	print(f"\n================    Model    {i}     ================\n")
	print("\n================ FOUND DENSE KERNELS ================\n")
	for path, arr in kernels.items():
		print(f"{path}")
		print(f"  shape: {arr.shape}")

	print("\n================ EXPECTED SHAPES ====================\n")
	for s in expected_kernels:
		print(" ", s)

	# # ------------------------------------------------------------
	# # CONSISTENCY CHECK
	# # ------------------------------------------------------------
	missing = [s for s in expected_kernels if s not in found_shapes]
	extra = [s for s in found_shapes if s not in expected_kernels]

	print("\n================ CONSISTENCY CHECK ==================\n")

	if not missing and not extra:
		print("✅ Weights are structurally consistent with meta_data.json")
	else:
		if missing:
			print("❌ Missing expected kernels:")
			for s in missing:
				print(" ", s)
		if extra:
			print("❌ Unexpected kernels found:")
			for s in extra:
				print(" ", s)

	print("\nDone.\n")

	# ------------------------------------------------------------
	# META DATA SUMMARY
	# ------------------------------------------------------------
	print("Meta data summary:")
	print(f"  Features         : {len(features)}")
	print(f"  Width layer 1    : {w1}")
	print(f"  Width layer 2    : {w2}")
	print(f"  Latent size      : {latent}")
	print(f"  Width last layer : {w_last}")
	print(f"  Output size      : {out}")
