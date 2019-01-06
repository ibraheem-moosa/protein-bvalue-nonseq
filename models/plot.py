import matplotlib.pyplot as plt
import numpy as np
import sys

for i in range(1, len(sys.argv)):
    fname = sys.argv[i]
    legend_name = fname.split('/')[0]
    print(legend_name)
    f = open(fname)
    l = [list(map(float, l.split())) for l in f]
    x = [2 * l[0] + 1 for l in l]
    y = [l[1] for l in l] 
    x = x[3:]
    y = y[3:]
    plt.plot(x, y, '+', lw=1, label=legend_name)
    plt.legend(loc='bottom right')
    #plt.legend(legend_name)
    f.close()

xlabel = "Window Size"
ylabel = "Average PCC on Validation Set"
plt.xlabel(xlabel)
plt.ylabel(ylabel)
#amax = np.argmax(y)
#xlim,ylim = plt.xlim(), plt.ylim()
#plt.plot([x[amax], x[amax], xlim[0]], [ylim[0], y[amax], y[amax]], linestyle="--")
#plt.xlim(xlim)
#plt.ylim(ylim)
plt.xticks(np.arange(7, 33, 2))
#plt.yticks(np.arange(0.21, 0.49, 0.02))
#print(amax, x[amax], y[amax])
plt.show()
