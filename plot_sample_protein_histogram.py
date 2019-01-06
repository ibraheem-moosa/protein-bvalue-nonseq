import os
import sys
import numpy as np
import matplotlib.pyplot as plt

fnames = os.listdir(sys.argv[1])
bvals = []
for fname in fnames:
    with open(os.path.join(sys.argv[1], fname)) as f:
        l = [l for l in f]
        l = [float(s.split()[1]) for s in l]
        l = np.array(l)
        l -= np.mean(l)
        l /= np.std(l)
        bvals.extend(list(l))

plt.hist(bvals, bins=200, density=True)
plt.show()


