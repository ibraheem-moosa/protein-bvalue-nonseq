import sys
import os
import random
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr
from Bio.PDB import Polypeptide

def aa_to_index(aa):
    """
    :param aa: Three character amino acid name.
    :returns: Integer index as per BioPython, unknown/non-standard amino acids return 20.
    """
    if Polypeptide.is_aa(aa, standard=True):
        return Polypeptide.three_to_index(aa)
    else:
        return 20

def one_hot(v, n=21):
    ret = np.zeros(n)
    ret[v] = 1
    return ret


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python3 linreg.py protein_list input_dir window_size')
        exit()

    protein_list_file = sys.argv[1]
    input_dir = sys.argv[2]
    #output_dir = sys.argv[3]
    ws = int(sys.argv[3])

    protein_list = []
    with open(protein_list_file) as f:
        protein_list.extend([l for l in f])


    indices = list(range(len(protein_list)))
    random.seed(42)
    random.shuffle(indices)
    train_indices = indices[:int(0.8 * len(indices))]
    validation_indices = indices[int(0.8 * len(indices)):]

    protein_seqs = []
    protein_bvals = []
    for protein in protein_list:
        protein = protein.strip()
        with open(os.path.join(input_dir, protein)) as f:
            lines = [l.split() for l in f]
            a = [aa_to_index(l[0]) for l in lines]
            b = [float(l[1]) for l in lines]
            protein_seqs.append(a)
            protein_bvals.append(b)

    print("Input read.")
    X = []
    y = []
    for i in train_indices:
        for j in range(ws, len(protein_seqs[i]) - ws):
            X.append(np.array(protein_seqs[i][j - ws:j + ws + 1]))
        y.extend(protein_bvals[i][ws:len(protein_bvals[i])-ws])

    X = np.vstack(X)
    y = np.array(y)

    oh = OneHotEncoder()
    oh.fit(X)
    X = oh.transform(X)
    print("Converted to numpy array.")
    clf = KNeighborsRegressor(n_jobs=2)
    clf.fit(X, y)
    
    print("Model fit done.")
    val_pccs = []
    for i in validation_indices:
        X = []
        for j in range(ws, len(protein_seqs[i]) - ws):
            X.append(np.array(protein_seqs[i][j - ws:j + ws + 1]))
        X = np.vstack(X)
        X = oh.transform(X)
        y_pred = clf.predict(X)
        val_pccs.append(pearsonr(y_pred, protein_bvals[i][ws:len(protein_bvals[i])-ws])[0])

    print("Validation Mean PCC: {}".format(sum(val_pccs) / len(val_pccs)))

