import matplotlib.pyplot as plt
import numpy as np
f = open('val_pccs_vs_window')
l = [float(l.split()[0]) for l in f]
x = np.arange(11, 2* 21 + 1, 2)##load x,y here
y = np.array(l)[5:21]## np.loadtxt()
xlabel = "Window Size"
ylabel = "Average PCC on Validation Set"
plt.plot(x, y, '+', lw=1)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
amax = np.argmax(y)
xlim,ylim = plt.xlim(), plt.ylim()
plt.plot([x[amax], x[amax], xlim[0]], [ylim[0], y[amax], y[amax]], linestyle="--")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xticks(np.arange(11, 45, 2))
plt.yticks(np.arange(0.464, 0.4690, 0.0005))
print(amax, x[amax], y[amax])
plt.show()
