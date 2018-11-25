import sys
import os
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
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

def get_pccs_and_mses(protein_seqs, protein_bvals, indices, ws, clf):
    pccs = []
    mses = []
    for i in indices:
        X = []
        for j in range(ws, len(protein_seqs[i]) - ws):
            X.append(np.array(protein_seqs[i][j - ws:j + ws + 1]))
        X = np.vstack(X)
        #X = oh.transform(X)
        y_pred = clf.predict(X)
        pccs.append(pearsonr(y_pred, protein_bvals[i])[0])
        mses.append(mean_squared_error(y_pred, protein_bvals[i]))

    pccs = np.array(pccs)
    mses = np.array(mses)
    return pccs, mses

def get_stats_on_pccs_and_mses(pccs, mses, prefix, ws, indices, protein_seqs, protein_bvals, protein_list):
    lens = [len(protein_seqs[i]) - 2 * ws for i in indices]
    bvals_mean = [protein_bvals[i].mean() for i in indices]
    plt.hist(pccs, density=True, range=(-0.5,1.0), bins=50)
    plt.savefig(prefix + '-pcc-hist-{:02d}.png'.format(ws))
    plt.close()
    plt.hist(mses, density=True, range=(0.0, 5.0), bins=50)
    plt.savefig(prefix + '-mse-hist-{:02d}.png'.format(ws))
    plt.close()
    print(prefix.upper())
    print("Mean PCC: {} +- {}".format(pccs.mean(), 3 * pccs.std()))
    print("PCC: Min: {} Max: {}".format(pccs.min(), pccs.max()))
    print("Mean MSE: {} +- {}".format(mses.mean(), 3 * mses.std()))
    print("MSE: Min: {} Max: {}".format(mses.min(), mses.max()))
 
    indices_sorted = sorted(enumerate(indices), key=lambda t: mses[t[0]])
    indices_sorted = [(protein_list[t[1]],len(protein_seqs[t[1]]) - 2 * ws) for t in indices_sorted]
    print("MSE:\nArgMin: {}\nArgMax: {}".format(indices_sorted[:10], list(reversed(indices_sorted[-10:]))))

    indices_sorted = sorted(enumerate(indices), key=lambda t: pccs[t[0]])
    indices_sorted = [(protein_list[t[1]],len(protein_seqs[t[1]]) - 2 * ws) for t in indices_sorted]
    print("PCC:\nArgMin: {}\nArgMax: {}".format(indices_sorted[:10], list(reversed(indices_sorted[-10:]))))


    clf = LinearRegression()
    clf.fit(mses.reshape((-1, 1)), pccs)
    print("MSE vs PCC correlation: {}".format(clf.score(mses.reshape((-1,1)), pccs)))
    print(clf.coef_)
    print(clf.intercept_)
    clf.fit(mses.reshape((-1, 1)), lens)
    print("MSE vs Length correlation: {}".format(clf.score(mses.reshape((-1,1)), lens)))
    clf.fit(pccs.reshape((-1, 1)), lens)
    print("PCC vs Length correlation: {}".format(clf.score(pccs.reshape((-1,1)), lens)))
    clf.fit(mses.reshape((-1, 1)), bvals_mean)
    print("MSE vs b-val mean correlation: {}".format(clf.score(mses.reshape((-1,1)), bvals_mean)))
    clf.fit(pccs.reshape((-1, 1)), bvals_mean)
    print("PCC vs b-val mean correlation: {}".format(clf.score(pccs.reshape((-1,1)), bvals_mean)))
 
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
        protein_list.extend([l.strip() for l in f])

    indices = list(range(len(protein_list)))
    random.seed(42)
    random.shuffle(indices)
    train_indices = indices[:int(0.8 * len(indices))]
    val_indices = train_indices[int(0.8 * len(train_indices)):]
    train_indices = train_indices[:int(0.8 * len(train_indices))]
    test_indices = indices[int(0.8 * len(indices)):]

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
            scaler = StandardScaler()
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

    print("Converted to numpy array.")

    clf = DecisionTreeRegressor(max_depth=10, random_state=42)
    clf.fit(X, y)
    print("Model fit done.")
    fi = clf.feature_importances_
    fi /= fi.max()
    print(fi)
    train_pccs, train_mses = get_pccs_and_mses(protein_seqs, protein_bvals, train_indices, ws, clf)
    val_pccs, val_mses = get_pccs_and_mses(protein_seqs, protein_bvals, val_indices, ws, clf)
    test_pccs, test_mses = get_pccs_and_mses(protein_seqs, protein_bvals, test_indices, ws, clf)
    get_stats_on_pccs_and_mses(train_pccs, train_mses, 'train', ws, train_indices, protein_seqs, protein_bvals, protein_list)
    get_stats_on_pccs_and_mses(val_pccs, val_mses, 'val', ws, val_indices, protein_seqs, protein_bvals, protein_list)
    get_stats_on_pccs_and_mses(test_pccs, test_mses, 'test', ws, test_indices, protein_seqs, protein_bvals, protein_list)

