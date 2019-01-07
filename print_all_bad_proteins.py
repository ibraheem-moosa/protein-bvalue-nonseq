import os
import sys
import random
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

protein_list_file = sys.argv[3]
protein_list = []
with open(protein_list_file) as f:
    protein_list.extend([l.strip() for l in f])

indices = list(range(len(protein_list)))
random.seed(42)
random.shuffle(indices)
train_indices = indices[:int(0.8 * len(indices))]
val_indices = train_indices[int(0.8 * len(train_indices)):]
train_indices = train_indices[:int(0.8 * len(train_indices))]
test_indices = indices[int(0.8 * len(indices)):]
train_set_proteins = set([protein_list[i].upper() for i in train_indices])
val_set_proteins = set([protein_list[i].upper() for i in val_indices])
test_set_proteins = set([protein_list[i].upper() for i in test_indices])

fnames = os.listdir(sys.argv[1])
bvals = dict()
for fname in fnames:
    with open(os.path.join(sys.argv[1], fname)) as f:
        l = [l for l in f]
        l = [float(s.split()[1]) for s in l]
        l = np.array(l)
        l -= np.mean(l)
        l /= np.std(l)
        bvals[fname.upper()] = list(l)

pred_fnames = os.listdir(sys.argv[2])
preds = dict()
for fname in pred_fnames:
    with open(os.path.join(sys.argv[2], fname)) as f:
        l = [l for l in f]
        l = [float(s) for s in l]
        l = np.array(l)
        l -= np.mean(l)
        l /= np.std(l)
        preds[fname.upper()] = list(l)

pccs = dict()
for key in preds:
    bval = bvals[key]
    pred = preds[key]
    pccs[key] = pearsonr(bval, pred)[0]
    if pccs[key] < 0.30:
        print(key)

pccs = list(pccs.values())
print(len(pccs))
print(max(pccs))
print(min(pccs))
print(sum(pccs)/len(pccs))
#plt.hist(pccs, bins=16, density=True)
#plt.show()


