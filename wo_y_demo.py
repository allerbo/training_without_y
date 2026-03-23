import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from help_fcts import predict
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams.update({'pdf.fonttype': 42, 'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman']})
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

lines=[Line2D([0],[0],color='C7',lw=2),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C0',lw=2),Line2D([0],[0],color='C1',lw=2),Line2D([0],[0],color='C2',lw=2),Line2D([0],[0],color='C3',lw=2),Line2D([0],[0],color='C4',lw=2)]
plt.cla()

labs = ['True Function', 'Noisy Observations', 'KRR', 'kNN', 'RF', 'NN', 'Smoothing Spline']


fig, axs=plt.subplots(2,1,figsize=(5.9,4.3))
for ax, apdx, title in zip(axs, ['_y','_s'],['Training with y', 'Training without y']):
  
  fh_nnr= predict(x_tr,x_te, x_te, y_tr, 'nnr'+apdx, DIM_H=20, dt=1e-2, gamma=0.95, max_epoch=20000)
  fh_spl= predict(x_tr, x_te, x_te, y_tr, 'spl'+apdx)
  fh_krr= predict(x_tr, x_te, x_te, y_tr, 'krr'+apdx)
  fh_knn= predict(x_tr, x_te, x_te, y_tr, 'knnr'+apdx)
  fh_rfr= predict(x_tr, x_te, x_te, y_tr, 'rfr'+apdx)
  
  _=ax.plot(x_te,y_te,'C7', lw=3)
  _=ax.plot(x_te,fh_krr,'C0')
  _=ax.plot(x_te,fh_knn,'C1')
  _=ax.plot(x_te,fh_rfr,'C2')
  _=ax.plot(x_te,fh_nnr,'C3')
  _=ax.plot(x_te,fh_spl,'C4')
  _=ax.plot(x_tr,y_tr,'ok')
  _=ax.set_xticks([])
  _=ax.set_yticks([])
  _=ax.set_ylim([-1.6,1.6])
  _=ax.set_title(title)
  
  
  fig.legend(lines, labs, loc='lower center', ncol=4)
  fig.tight_layout()
  fig.subplots_adjust(bottom=.14)
  fig.savefig('figures/wo_y_demo.pdf')

