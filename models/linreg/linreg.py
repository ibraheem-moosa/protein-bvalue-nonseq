import sys
import os
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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
            a = ws * [20]
            a.extend([aa_to_index(l[0]) for l in lines])
            a.extend(ws * [20])
            b = [float(l[1]) for l in lines]
            b = np.array(b)
            scaler = RobustScaler()
            b = scaler.fit_transform(b.reshape((-1, 1))).reshape((-1,))
            protein_seqs.append(a)
            protein_bvals.append(b)

    print("Input read.")
    X = []
    y = []
    for i in train_indices:
        for j in range(ws, len(protein_seqs[i]) - ws):
            X.append(np.array(protein_seqs[i][j - ws:j + ws + 1]))
        y.extend(protein_bvals[i])

    X = np.vstack(X)
    y = np.array(y)

    oh = OneHotEncoder()
    oh.fit(X)
    X = oh.transform(X)
    print("Converted to numpy array.")
    clf = LinearRegression()
    clf.fit(X, y)
    
    print("Model fit done.")
    val_pccs = []
    val_mses = []
    for i in validation_indices:
        X = []
        for j in range(ws, len(protein_seqs[i]) - ws):
            X.append(np.array(protein_seqs[i][j - ws:j + ws + 1]))
        X = np.vstack(X)
        X = oh.transform(X)
        y_pred = clf.predict(X)
        val_pccs.append(pearsonr(y_pred, protein_bvals[i])[0])
        val_mses.append(mean_squared_error(y_pred, protein_bvals[i]))

    val_pccs = np.array(val_pccs)
    val_mses = np.array(val_mses)
    plt.hist(val_pccs, density=True, range=(-0.5,1.0), bins=25)
    plt.savefig('pcc-hist-{:02d}.png'.format(ws))
    plt.close()
    plt.hist(val_mses, density=True, range=(0.0, 5.0), bins=50)
    plt.savefig('mse-hist-{:02d}.png'.format(ws))

    print("Validation Mean PCC: {} +- {}".format(val_pccs.mean(), 3 * val_pccs.std()))
    print("Validation PCC: Min: {} Max: {}".format(val_pccs.min(), val_pccs.max()))
    val_pccs_argsorted = val_pccs.argsort()
    print("Validation PCC:\nArgMin: {}\nArgMax: {}".format(val_pccs_argsorted[:10], val_pccs_argsorted[-10:]))

