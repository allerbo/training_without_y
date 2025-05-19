import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from help_fcts import kern, make_spline, ssmm_loocv, ssmm_S, loogcv_S
from nnr_fcts import init_model, train_step, get_K, get_S
import os
os.environ['JAX_ENABLE_X64']='True'
from jax import config as jc
jc.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

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

lines=[Line2D([0],[0],color='C7',lw=2),plt.plot(0,0,'ok')[0],Line2D([0],[0],color='C0',lw=2),Line2D([0],[0],color='C1',lw=2),Line2D([0],[0],color='C2',lw=2)]
plt.cla()

labs = ['True Function', 'Noisy Observations', 'Kernel Ridge Regression', 'Smoothing Spline', 'Neural Network']

I_tr=np.eye(n_tr)

lbdas_cv=np.hstack(([1e-10],np.geomspace(1e-4,1,100)))
sigmas_cv=np.geomspace(1e-4,1,100)


lbda_krr_n, sigma_krr_n=ssmm_loocv(x_tr,x_te,lbdas_cv,sigmas_cv)
K_tr_n=kern(x_tr,x_tr,sigma_krr_n)
K_te_n=kern(x_te,x_tr,sigma_krr_n)
fh_krr_n=K_te_n@np.linalg.solve(K_tr_n+lbda_krr_n*I_tr,y_tr)
lbda_krr_y, sigma_krr_y=ssmm_loocv(x_tr,x_te,lbdas_cv,sigmas_cv,y_tr)
K_tr_y=kern(x_tr,x_tr,sigma_krr_y)
K_te_y=kern(x_te,x_tr,sigma_krr_y)
fh_krr_y=K_te_y@np.linalg.solve(K_tr_y+lbda_krr_y*I_tr,y_tr)

B_tr,B_te,Omega=make_spline(x_tr,x_te)
lbda_sp_n,_=ssmm_loocv(x_tr,x_te,lbdas_cv)
fh_sp_n=B_te@np.linalg.solve(B_tr.T@B_tr+lbda_sp_n*Omega+1e-8*np.eye(B_tr.shape[1]),B_tr.T@y_tr)
lbda_sp_y,_=ssmm_loocv(x_tr,x_te,lbdas_cv,y_tr=y_tr)
fh_sp_y=B_te@np.linalg.solve(B_tr.T@B_tr+lbda_sp_y*Omega+1e-8*np.eye(B_tr.shape[1]),B_tr.T@y_tr)

dt=1e-2
gamma=0.95

model_state = init_model(1, 20, 1, dt, gamma)

S_tr=jnp.zeros((n_tr,n_tr))
S_val=jnp.zeros((n_tr,n_tr))
S_te=jnp.zeros((n_te,n_tr))

S_tr_old=jnp.copy(S_tr)
S_val_old=jnp.copy(S_val)
S_te_old=jnp.copy(S_te)

best_mse=np.inf
fh_nnr_y=None
for epoch in range(10001):
  if epoch % 100==0:
    mse=loogcv_S(S_tr,y_tr)
    if epoch>200 and mse<best_mse:
      best_mse=mse
      fh_nnr_y=S_te@y_tr
    K_tr, K_val, K_te=get_K(x_tr,x_tr,x_te,model_state)
  model_state = train_step(model_state, x_tr, y_tr)
  S_tr, S_tr_old, S_val, S_val_old, S_te, S_te_old = get_S(K_tr,K_val,K_te,S_tr, S_val,S_te,S_tr_old,S_val_old,S_te_old,dt,gamma)

fh_nnr_y=S_te@y_tr

y_trr=np.random.normal(0,np.std(y_tr),y_tr.shape)

model_state = init_model(1, 20, 1, dt, gamma)

S_tr=jnp.zeros((n_tr,n_tr))
S_val=jnp.zeros((n_tr,n_tr))
S_te=jnp.zeros((n_te,n_tr))

S_tr_old=jnp.copy(S_tr)
S_val_old=jnp.copy(S_val)
S_te_old=jnp.copy(S_te)

best_mse=np.inf
fh_nnr_n=None
for epoch in range(20001):
  if epoch % 100==0:
    mse=ssmm_S(S_te)
    if mse<best_mse:
      best_mse=mse
      fh_nnr_n=S_te@y_tr
    K_tr, K_val, K_te=get_K(x_tr,x_tr,x_te,model_state)
  model_state = train_step(model_state, x_tr, y_trr)
  S_tr, S_tr_old, S_val, S_val_old, S_te, S_te_old = get_S(K_tr,K_val,K_te,S_tr, S_val,S_te,S_tr_old,S_val_old,S_te_old,dt,gamma)


fig, axs=plt.subplots(2,1,figsize=(7,3))
for ax, fh_krr, fh_sp, fh_nnr, title in zip(axs,[fh_krr_y,fh_krr_n],[fh_sp_y,fh_sp_n],[fh_nnr_y,fh_nnr_n],['Training with y', 'Training without y']):
  _=ax.plot(x_te,y_te,'C7', lw=2)
  _=ax.plot(x_te,fh_krr,'C0')
  _=ax.plot(x_te,fh_sp,'C1')
  _=ax.plot(x_te,fh_nnr,'C2')
  _=ax.plot(x_tr,y_tr,'ok')
  _=ax.set_xticks([])
  _=ax.set_yticks([])
  _=ax.set_ylim([-1.6,1.6])
  _=ax.set_title(title)


fig.legend(lines, labs, loc='lower center', ncol=3)
fig.tight_layout()
fig.subplots_adjust(bottom=.19)
fig.savefig('figures/wo_y_demo.pdf')
