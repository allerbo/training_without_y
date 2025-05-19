import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from help_fcts import kern

np.random.seed(3)


n_tr=10
n_te=1000

x_tr=np.sort(np.random.uniform(-1,1,n_tr)).reshape(-1,1)
x_tr[0]=-1
x_tr[-1]=1
x_te=np.linspace(-1,1,n_te).reshape(-1,1)

np.random.seed(3)
y_tr=np.sin(2*np.pi*x_tr)+np.random.normal(0,.3,x_tr.shape)
y_te=np.sin(2*np.pi*x_te)

lines=[Line2D([0],[0],color='C7',lw=2),plt.plot(0,0,'ok')[0]]
lines1=[Line2D([0],[0],color='C7',lw=2),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C3',lw=2),Line2D([0],[0],color='C2',lw=2),Line2D([0],[0],color='C1',lw=2)]
plt.cla()

labs = ['True Function', 'Noisy Observations']

I_tr=np.eye(n_tr)


fig, ax=plt.subplots(1,1,figsize=(7,2))
fig1, ax1=plt.subplots(1,1,figsize=(7,2.5))
_=ax.plot(x_te,y_te,'C7', lw=2)
_=ax.plot(x_tr,y_tr,'ok')
_=ax1.plot(x_te,y_te,'C7', lw=2)
_=ax1.plot(x_tr,y_tr,'ok')
for lbda,sigma,cc in zip([0,0,0], [1.3,.16,.01], ['C3','C2', 'C1']):
  K_tr=kern(x_tr,x_tr,sigma)
  K_te=kern(x_te,x_tr,sigma)
  _=ax.plot(x_te,K_te@np.linalg.solve(K_tr+lbda*I_tr,y_tr),cc)
  _=ax1.plot(x_te,K_te@np.linalg.solve(K_tr+lbda*I_tr,y_tr),cc)
  S_tr=K_tr@np.linalg.inv(K_tr+lbda*I_tr)
  S_te=K_te@np.linalg.inv(K_tr+lbda*I_tr)


_=ax.set_xticks([])
_=ax.set_yticks([])
_=ax.set_ylim([-1.4,1.6])

_=ax1.set_xticks([])
_=ax1.set_yticks([])
_=ax1.set_ylim([-1.4,1.6])

fig.legend(lines, labs, loc='lower center', ncol=2)
fig.tight_layout()
fig.subplots_adjust(bottom=.18)
fig.savefig('figures/compl_demo.pdf')

