import sys

protein_list_fname = sys.argv[1]
bad_proteins_fname = sys.argv[2]

protein_list = []

with open(protein_list_fname) as f:
    protein_list = [l.strip().lower() for l in f]

bad_proteins = []

with open(bad_proteins_fname) as f:
    bad_proteins = [l.strip().lower() for l in f]

good_proteins = sorted(list(set(protein_list).difference(bad_proteins)))

for p in good_proteins:
    print(p)
