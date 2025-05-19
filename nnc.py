import numpy as np
from matplotlib import pyplot as plt
import time
from help_fcts import acc, make_real_data, loogcv_S, ssmm_S
from nnc_fcts import init_model, train_step, get_K, get_S, get_Ft
import os, sys
os.environ['JAX_ENABLE_X64']='True'
from jax import config as jc
jc.update('jax_platform_name', 'cpu')
import jax.numpy as jnp



ALGS=['ssmm_n', 'loocv_n']

def ssmm_n(S_tr,S_val,y_tr,y_tr_r):
  return ssmm_S(S_val.reshape(S_val.shape[0]*S_val.shape[1],S_val.shape[2]*S_val.shape[3]))

def loocv_n(S_tr,S_val,y_tr,y_tr_r):
  return loogcv_S(S_tr.reshape(S_tr.shape[0]*S_tr.shape[1],S_tr.shape[2]*S_tr.shape[3]))



dt=1e-4
gamma=0.7
N_TR=500
DIM_H=200

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

for data in ['mnist','cifar']:
  acc_trs={}
  acc_tes={}
  stop_times={}
  for alg in ALGS:
    acc_trs[alg]=[]
    acc_tes[alg]=[]
    stop_times[alg]=[]
  
  for seed in range(10):
    acc_trs_seed=[]
    acc_tes_seed=[]
    epochs_seed=[]
    errors_seed={}
    for alg in ALGS:
      errors_seed[alg]=[]
    
    X_tr, y_tr, X_te, y_te, X_val=make_real_data(data, seed, N_TR)
    n_tr,p=X_tr.shape
    n_val=X_val.shape[0]
    n_te=X_te.shape[0]
    dim_y=y_tr.shape[1]
    y_tr_r=np.eye(dim_y)[np.random.randint(0,dim_y,n_tr)]
    
    model_state = init_model(p,DIM_H,dim_y, dt, gamma)
    
    fh_tr0=model_state.apply_fn(model_state.params,X_tr)
    fh_val0=model_state.apply_fn(model_state.params,X_val)
    fh_te0=model_state.apply_fn(model_state.params,X_te)
    
    S_tr=jnp.zeros((n_tr,dim_y-1,n_tr,dim_y-1))
    S_val=jnp.zeros((n_val,dim_y-1,n_tr,dim_y-1))
    S_te=jnp.zeros((n_te,dim_y-1,n_tr,dim_y-1))
    
    S_tr_old=jnp.copy(S_tr)
    S_val_old=jnp.copy(S_val)
    S_te_old=jnp.copy(S_te)
    
    
    
    for epoch in range(201):
      if epoch % 5==0:
        acc_trs_seed.append(acc(y_tr, jnp.einsum('ijkl,kl',S_tr,(y_tr-fh_tr0)[:,:-1])+fh_tr0[:,:-1]))
        acc_tes_seed.append(acc(y_te, jnp.einsum('ijkl,kl',S_te,(y_tr-fh_tr0)[:,:-1])+fh_te0[:,:-1]))
        epochs_seed.append(epoch)
        for alg in ALGS:
          errors_seed[alg].append(eval(alg)(S_tr,S_val,y_tr,y_tr_r))
      if epoch % 20 == 0:
        K_tr, K_val, K_te=get_K(X_tr,X_val,X_te,model_state)
        Ft=get_Ft(X_tr,model_state)
      model_state = train_step(model_state, X_tr, y_tr_r)
      S_tr, S_tr_old, S_val, S_val_old, S_te, S_te_old = get_S(K_tr,K_val,K_te,S_tr, S_val,S_te,S_tr_old,S_val_old,S_te_old,Ft,dt,gamma)
      if jnp.isnan(jnp.sum(S_tr)):
        break
    
    for alg in ALGS:
      acc_trs[alg].append(acc_trs_seed[np.argmin(errors_seed[alg])])
      acc_tes[alg].append(acc_tes_seed[np.argmin(errors_seed[alg])])
      stop_times[alg].append(epochs_seed[np.argmin(errors_seed[alg])])
  for ii, alg in enumerate(ALGS):
    if ii==0:
      print(('\\multirow{'+str(len(ALGS))+'}{*}{'+data+'}').ljust(22),end='')
    else:
      print(''.ljust(22),end='')
    print('& '+alg.ljust(9),end='')
    print(f'& {np.median(acc_tes[alg]):.2f} ({np.quantile(acc_tes[alg],0.25):.2f}, {np.quantile(acc_tes[alg],0.75):.2f}) '.ljust(23), end='')
    print(f'& {np.median(stop_times[alg]):.0f} ({np.quantile(stop_times[alg],0.25):.0f}, {np.quantile(stop_times[alg],0.75):.0f})'.ljust(20)+'\\\\')
  print('\\hline')
print('')
