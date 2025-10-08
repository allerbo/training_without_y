import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

n=100
sig=0.25

cl0=np.random.normal(0,sig,(n,2))+np.array((-1,1))
cl1=np.random.normal(0,sig,(n,2))+np.array((1,1))
cl2=np.random.normal(0,sig,(n,2))+np.array((-1,-1))
cl3=np.random.normal(0,sig,(n,2))+np.array((1,-1))

cs=[['oC0','oC1','oC2','oC3'], ['ok','ok','ok','ok'], ['ok','ok','ok','ok'], ['oC0','oC1','oC2','oC3']]
titles=['a','b','c','d']

fig, axs=plt.subplots(2,2,figsize=(5,5))
for ii, (ax, c, title) in enumerate(zip(axs.ravel(),cs, titles)):
  _=ax.set_title(title)
  _=ax.plot(cl0[:,0],cl0[:,1],c[0],ms=3)
  _=ax.plot(cl1[:,0],cl1[:,1],c[1],ms=3)
  _=ax.plot(cl2[:,0],cl2[:,1],c[2],ms=3)
  _=ax.plot(cl3[:,0],cl3[:,1],c[3],ms=3)
  ax.set_xlim([-2,2])
  ax.set_ylim([-2,2])
  ax.axis('off')
  if ii>=2:
    _=ax.axvline(0,color='C4')
    _=ax.axhline(0,color='C4')
  if ii==3:
    _=ax.fill_between((-2.01,0), (2.01,2.01),facecolor='C0',alpha=0.3)
    _=ax.fill_between((0,2.01), (2.01,2.01),facecolor='C1',alpha=0.3)
    _=ax.fill_between((-2.01,0), (-2.01,-2.01),facecolor='C2',alpha=0.3)
    _=ax.fill_between((0,2.01), (-2.01,-2.01),facecolor='C3',alpha=0.3)

fig.add_artist(plt.Line2D([0,1],[0.5,0.5], transform=fig.transFigure,color='black'))
fig.add_artist(plt.Line2D([0.5,0.5],[0,1], transform=fig.transFigure,color='black'))

plt.tight_layout()
fig.savefig('figures/cl_demo.pdf')
