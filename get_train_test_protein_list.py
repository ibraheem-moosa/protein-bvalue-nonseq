import os
import sys
import random
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

protein_list_file = sys.argv[1]
protein_list = []
with open(protein_list_file) as f:
    protein_list.extend([l.strip() for l in f])

indices = list(range(len(protein_list)))
random.seed(42)
random.shuffle(indices)
train_indices = indices[:int(0.8 * len(indices))]
#val_indices = train_indices[int(0.8 * len(train_indices)):]
#train_indices = train_indices[:int(0.8 * len(train_indices))]
test_indices = indices[int(0.8 * len(indices)):]
train_set_proteins = set([protein_list[i] for i in train_indices])
#val_set_proteins = set([protein_list[i] for i in val_indices])
test_set_proteins = set([protein_list[i] for i in test_indices])

with open('train_protein_list', 'w') as f:
    for p in train_set_proteins:
        f.write('{}\n'.format(p))

with open('test_protein_list', 'w') as f:
    for p in test_set_proteins:
        f.write('{}\n'.format(p))


