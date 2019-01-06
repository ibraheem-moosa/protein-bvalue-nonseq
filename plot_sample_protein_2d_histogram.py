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
        l = list(l)
        length = len(l) - 1
        l = [(i / length, l[i]) for i in range(len(l))]
        bvals.extend(l)

y = [s[0] for s in l]
x = [s[1] for s in l]
plt.hist2d(x, y, bins=16, normed=True)
plt.show()


