import sys
import os

keys_file = sys.argv[1]
cif_dir = sys.argv[2]
out_path = sys.argv[3]

keys = []
with open(keys_file) as f:
    for l in f:
        keys.append(l.strip())

for k in keys:
    print(k)

out_dict = dict()
cif_files = os.listdir(cif_dir)
cif_files = [os.path.join(cif_dir, fname) for fname in cif_files]
miss_dict = dict()

curr_protein = 0
for fname in cif_files:
    print("{} Extracting {}".format(curr_protein, fname))
    curr_protein += 1
    protein = fname[4:-4]
    out_dict[protein] = dict()
    with open(fname) as f:
        for l in f:
            l = l.strip()
            for k in keys:
                never_found = True
                if l.startswith(k):
                    value = l[len(k):].strip().upper()
                    out_dict[protein][k] = value
                    never_found = False
                if never_found:
                    if k in miss_dict:
                        miss_dict[k] += 1
                    else:
                        miss_dict[k] = 1
                    out_dict[protein][k] = '?'
                    #print('{} not found in {}'.format(k, protein))

for k in miss_dict:
    print(k, miss_dict[k])

with open(out_path, "w") as f:
    out_line = 'protein,'
    for k in keys[:-1]:
        out_line += k + ','
    out_line += keys[-1] + '\n'
    f.write(out_line)
    for fname in cif_files:
        protein = fname[4:-4]
        out_line = protein + ','
        for k in keys[:-1]:
            out_line += out_dict[protein][k] + ','
        out_line += out_dict[protein][keys[-1]] + '\n'
        f.write(out_line)

