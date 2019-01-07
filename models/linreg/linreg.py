import sys
import os
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from Bio.PDB import Polypeptide
import pandas as pd
from sklearn.impute import SimpleImputer

def aa_to_index(aa):
    """
    :param aa: Three character amino acid name.
    :returns: Integer index as per BioPython, unknown/non-standard amino acids return 20.
    """
    if Polypeptide.is_aa(aa, standard=True):
        return Polypeptide.three_to_index(aa)
    else:
        return 20

def get_pccs_and_mses(protein_seqs, protein_bvals, protein_mdatas, indices, ws, clf, oh, imp, use_metadata=False):
    pccs = []
    mses = []
    preds = []
    for i in indices:
        X = []
        for j in range(ws, len(protein_seqs[i]) - ws):
            if use_metadata:
                X.append(np.hstack([np.array(protein_seqs[i][j - ws:j + ws + 1]), protein_mdatas[i]]))
            else:
                X.append(np.array(protein_seqs[i][j - ws:j + ws + 1]))
        X = np.vstack(X)
        X = imp.transform(X)
        X = oh.transform(X)
        y_pred = clf.predict(X)
        pccs.append(pearsonr(y_pred, protein_bvals[i])[0])
        mses.append(mean_squared_error(y_pred, protein_bvals[i]))
        preds.append(y_pred)

    pccs = np.array(pccs)
    mses = np.array(mses)
    return pccs, mses, preds

def get_stats_on_pccs_and_mses(pccs, mses, prefix, ws, indices, protein_seqs, protein_bvals, protein_list):
    lens = [len(protein_seqs[i]) - 2 * ws for i in indices]
    bvals_mean = [protein_bvals[i].mean() for i in indices]
    '''
    plt.hist(pccs, density=True, range=(-0.5,1.0), bins=50)
    plt.savefig(prefix + '-pcc-hist-{:02d}.png'.format(ws))
    plt.close()
    plt.hist(mses, density=True, range=(0.0, 5.0), bins=50)
    plt.savefig(prefix + '-mse-hist-{:02d}.png'.format(ws))
    plt.close()
    '''
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
    clf.fit(mses.reshape((-1, 1)), lens)
    print("MSE vs Length correlation: {}".format(clf.score(mses.reshape((-1,1)), lens)))
    clf.fit(pccs.reshape((-1, 1)), lens)
    print("PCC vs Length correlation: {}".format(clf.score(pccs.reshape((-1,1)), lens)))
    clf.fit(mses.reshape((-1, 1)), bvals_mean)
    print("MSE vs b-val mean correlation: {}".format(clf.score(mses.reshape((-1,1)), bvals_mean)))
    clf.fit(pccs.reshape((-1, 1)), bvals_mean)
    print("PCC vs b-val mean correlation: {}".format(clf.score(pccs.reshape((-1,1)), bvals_mean)))


def write_preds(indices, protein_list, preds, dirname):
    for i in range(len(indices)):
        protein = protein_list[indices[i]]
        with open(os.path.join(dirname, protein), "w") as f:
            for j in range(len(preds[i])):
                f.write('{}\n'.format(preds[i][j]))

 
if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Usage: python3 linreg.py protein_list input_dir protein_metadata window_size output_dir bad_protein_list')
        exit()

    use_metadata = False
    #use_metadata = True

    protein_list_file = sys.argv[1]
    input_dir = sys.argv[2]
    #output_dir = sys.argv[3]
    protein_metadata = sys.argv[3]
    ws = int(sys.argv[4])
    pred_dir = sys.argv[5]
    bad_protein_list_fname = sys.argv[6]
    print(ws)

    protein_list = []
    with open(protein_list_file) as f:
        protein_list.extend([l.strip() for l in f])

    bad_protein_list = []
    with open(bad_protein_list_fname) as f:
        bad_protein_list.extend([l.strip().lower() for l in f])


    protein_metadata = pd.read_csv(protein_metadata, parse_dates=[1,11], na_values=['.'])
    protein_metadata = protein_metadata.dropna(axis=1, thresh=protein_metadata.shape[0] - 1500)
    protein_metadata = protein_metadata[['protein', #'_entity_src_gen.pdbx_gene_src_ncbi_taxonomy_id ', #'_entity_src_gen.pdbx_host_org_ncbi_taxonomy_id ', 
        '_exptl_crystal_grow.pH ', '_exptl_crystal_grow.temp ', '_exptl_crystal.density_percent_sol ', '_exptl_crystal.density_Matthews ',
        '_diffrn.ambient_temp ',
        #'_refine_hist.d_res_low ', '_refine_hist.d_res_high ',
        #'_refine.ls_d_res_high ', '_refine.ls_d_res_low ',
        '_reflns.d_resolution_low ', '_reflns.d_resolution_high ',
        '_refine.ls_R_factor_R_free ',
        '_refine.ls_R_factor_obs ',
        '_refine.ls_R_factor_R_work ',
        '_reflns.pdbx_redundancy ',
        '_refine_hist.pdbx_number_atoms_protein ', '_refine_hist.pdbx_number_atoms_ligand ', '_refine_hist.number_atoms_solvent '
        ]]
    protein_metadata['_refine_hist.number_atoms_total'] = protein_metadata['_refine_hist.number_atoms_solvent '] + protein_metadata['_refine_hist.pdbx_number_atoms_ligand '] + protein_metadata['_refine_hist.pdbx_number_atoms_protein ']
    protein_metadata['_refine_hist.pdbx_number_atoms_protein '] /= protein_metadata['_refine_hist.number_atoms_total']
    protein_metadata['_refine_hist.pdbx_number_atoms_ligand '] /= protein_metadata['_refine_hist.number_atoms_total']
    protein_metadata['_refine_hist.number_atoms_solvent '] /= protein_metadata['_refine_hist.number_atoms_total']
    protein_metadata.drop('_refine_hist.number_atoms_total', axis=1, inplace=True)
    #print(protein_metadata.select_dtypes(include=np.number).keys())
    num_of_mdatas = protein_metadata.shape[1] - 2

    indices = list(range(len(protein_list)))
    random.seed(42)
    random.shuffle(indices)
    train_indices = indices[:int(0.8 * len(indices))]
    val_indices = train_indices[int(0.8 * len(train_indices)):]
    train_indices = train_indices[:int(0.8 * len(train_indices))]
    test_indices = indices[int(0.8 * len(indices)):]

    bad_protein_indices = [protein_list.index(p) for p in bad_protein_list]
    train_indices = list(set(train_indices).difference(bad_protein_indices))
    val_indices = list(set(val_indices).difference(bad_protein_indices))
    test_indices = list(set(test_indices).difference(bad_protein_indices))

    protein_seqs = []
    protein_bvals = []
    protein_mdatas = []
    for protein in protein_list:
        protein = protein.strip()
        metadata = protein_metadata[protein_metadata['protein'] == protein].select_dtypes(include=np.number)
        metadata = metadata.values[0]
        protein_mdatas.append(metadata)
        with open(os.path.join(input_dir, protein)) as f:
            lines = [l.split() for l in f]
            a = ws * [20]
            a.extend([aa_to_index(l[0]) for l in lines])
            a.extend(ws * [20])
            b = [float(l[1]) for l in lines]
            b = np.array(b)
            scaler = StandardScaler()
            b = scaler.fit_transform(b.reshape((-1, 1))).reshape((-1,))
            #b = b.clip(min=-2.0, max=2.0)
            protein_seqs.append(a)
            protein_bvals.append(b)

    print("Input read.")
    X = []
    y = []
    for i in train_indices:
        for j in range(ws, len(protein_seqs[i]) - ws):
            if use_metadata:
                X.append(np.hstack([np.array(protein_seqs[i][j - ws:j + ws + 1]), protein_mdatas[i]]))
            else:
                X.append(np.array(protein_seqs[i][j - ws:j + ws + 1]))
        y.extend(protein_bvals[i])

    X = np.vstack(X)
    y = np.array(y)

    imp = SimpleImputer(strategy='most_frequent')
    imp.fit(X)
    X = imp.transform(X)

    categorical_features=list(range(2 * ws + 1))
    if use_metadata:
        #categorical_features.append(2 * ws + 1 + 0)
        #categorical_features.append(2 * ws + 1 + 1)
        pass
    print(X.shape)
    oh = OneHotEncoder(categorical_features=categorical_features)
    oh.fit(X)
    X = oh.transform(X)
    print(X.shape)
    print("Converted to numpy array.")
    clf = LinearRegression()
    clf.fit(X, y)
    print("Model fit done.")
    #print(clf.score(X, y))
    w = clf.coef_
    b = clf.intercept_
    if use_metadata:
        mw = w[21 * (2 * ws + 1):]
        pw = w[:21 * (2 * ws + 1)]
        pw = pw.reshape((-1, 21))
        pw = np.linalg.norm(pw, axis=1)
        print(mw)
        print(pw)
    else:
        w = w.reshape((-1, 21))
        w = np.linalg.norm(w, axis=1)
        print(w)
        print(b)
    train_pccs, train_mses, train_preds = get_pccs_and_mses(protein_seqs, protein_bvals, protein_mdatas, train_indices, ws, clf, oh, imp, use_metadata)
    val_pccs, val_mses, val_preds = get_pccs_and_mses(protein_seqs, protein_bvals, protein_mdatas, val_indices, ws, clf, oh, imp, use_metadata)
    test_pccs, test_mses, test_preds = get_pccs_and_mses(protein_seqs, protein_bvals, protein_mdatas, test_indices, ws, clf, oh, imp, use_metadata)
    get_stats_on_pccs_and_mses(train_pccs, train_mses, 'train', ws, train_indices, protein_seqs, protein_bvals, protein_list)
    get_stats_on_pccs_and_mses(val_pccs, val_mses, 'val', ws, val_indices, protein_seqs, protein_bvals, protein_list)
    get_stats_on_pccs_and_mses(test_pccs, test_mses, 'test', ws, test_indices, protein_seqs, protein_bvals, protein_list)
    os.makedirs(pred_dir, exist_ok=True)
    write_preds(train_indices, protein_list, train_preds, pred_dir)
    write_preds(val_indices, protein_list, val_preds, pred_dir)
    write_preds(test_indices, protein_list, test_preds, pred_dir)
