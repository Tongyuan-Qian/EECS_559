import numpy as np

from utils import *
import matplotlib.pyplot as plt
import sas_bd
from scipy.linalg import circulant

p0 = int(1e3)
n = int(1e5)
theta = 2/p0

a0 = nor(p0)
a0 /= norm(a0)

x0 = nor(n) * ber(n, theta)
y = c_conv(x0, a0, n)

solver = sas_bd.SasBd(theta, p0, y)
xh, ah = solver.solve()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
ax1.plot(a0, 'r', linewidth=0.5, label=r'$a_0$')
ax1.plot(ah, 'b', linewidth=0.5, label=r'$\hat{a}$')
ax1.legend(loc=1)

x0_m, x0_s, x0_b = ax2.stem(np.arange(n), x0, linefmt='r:', markerfmt='ro', label=r'$x_0$')
xh_m, xh_s, xh_b = ax2.stem(np.arange(n), xh, linefmt='b:', markerfmt='bo', label=r'$\hat{x}$')
plt.setp(x0_s, linewidth=0.5)
plt.setp(xh_s, linewidth=0.5)
plt.setp(x0_m, markersize=1)
plt.setp(xh_m, markersize=1)
plt.setp(x0_b, visible=False)
plt.setp(xh_b, visible=False)
ax2.legend(loc=1)

y0 = y
ax3.plot(y0, 'r', linewidth=0.5, label=r'$y_0$')
yh = c_conv(xh, ah, n)
ax3.plot(yh, 'b', linewidth=0.5, label=r'$\hat{y}$')
ax3.legend(loc=1)

plt.savefig("plot.png", dpi=1000)
plt.show()

a0 = np.pad(a0, (0, len(ah) - len(a0)), 'constant')
al = circulant(a0).T
res1 = np.max(al @ ah)
res2 = np.max(-al @ ah)
res = max(res2, res1)
print(res)






