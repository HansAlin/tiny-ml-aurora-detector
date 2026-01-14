import h5py
import numpy as np

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
WEIGHTS_1 = r"experiments\classifier_experiment_2\model_final.weights.h5"
WEIGHTS_2 = r"experiments\classifier_experiment_2\model_weights\Encoder_classifier_epoch_149.weights.h5"

ROW_DIFF_EPS = 1e-12  # threshold for printing row diffs

# ------------------------------------------------------------
# HDF5 UTILITIES
# ------------------------------------------------------------
def collect_tensors(h5file):
    tensors = {}

    def walk(group, prefix=""):
        for k in group.keys():
            item = group[k]
            path = f"{prefix}/{k}"
            if isinstance(item, h5py.Dataset):
                tensors[path] = np.array(item)
            else:
                walk(item, path)

    walk(h5file)
    return tensors


def is_dense_kernel(path, array):
    return path.endswith("/kernel") and array.ndim == 2


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
with h5py.File(WEIGHTS_1, "r") as f1, h5py.File(WEIGHTS_2, "r") as f2:

    t1 = collect_tensors(f1)
    t2 = collect_tensors(f2)

paths_1 = set(t1)
paths_2 = set(t2)

common = sorted(paths_1 & paths_2)
only_1 = sorted(paths_1 - paths_2)
only_2 = sorted(paths_2 - paths_1)

print("\n================ COMMON TENSORS ================\n")

for path in common:
    a = t1[path]
    b = t2[path]

    if a.shape != b.shape:
        print(f"[SHAPE MISMATCH] {path}: {a.shape} vs {b.shape}")
        continue

    diff = a - b
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))

    print(f"{path}")
    print(f"  shape     : {a.shape}")
    print(f"  max |Δ|   : {max_diff:.3e}")
    print(f"  mean |Δ|  : {mean_diff:.3e}")

    # Row-by-row diagnostics for Dense kernels
    if is_dense_kernel(path, a):
        row_norms = np.linalg.norm(diff, axis=1)
        bad_rows = np.where(row_norms > ROW_DIFF_EPS)[0]

        if len(bad_rows) > 0:
            print("  row diffs:")
            for i in bad_rows[:10]:  # cap output
                print(f"    row {i:4d}: ||Δ|| = {row_norms[i]:.3e}")
        else:
            print("  rows      : identical")

    print()

print("\n================ ONLY IN FILE 1 ================\n")
for p in only_1:
    print(p)

print("\n================ ONLY IN FILE 2 ================\n")
for p in only_2:
    print(p)

print("\nDone.")
