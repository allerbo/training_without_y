import numpy as np
from matplotlib import pyplot as plt
import time
from help_fcts import r2, make_real_data,  loogcv_S, ssmm_S
from nnr_fcts import init_model, train_step, get_K, get_S
import os, sys
os.environ['JAX_ENABLE_X64']='True'
from jax import config as jc
jc.update('jax_platform_name', 'cpu')
import jax.numpy as jnp



ALGS=['ssmm_n', 'loocv_n']

def loocv_n(S_tr,S_val,y_tr,y_tr_r):
  return loogcv_S(S_tr)

def ssmm_n(S_tr,S_val,y_tr,y_tr_r):
  return ssmm_S(S_val)


DIM_H=200
dt=1e-4
gamma=0.7
N_TR=500

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

for data in ['steel','cpu','super','power']:
  r2_trs={}
  r2_tes={}
  stop_times={}
  for alg in ALGS:
    r2_trs[alg]=[]
    r2_tes[alg]=[]
    stop_times[alg]=[]
  
  for seed in range(10):
    r2_trs_seed=[]
    r2_tes_seed=[]
    epochs_seed=[]
    errors_seed={}
    for alg in ALGS:
      errors_seed[alg]=[]
    X_tr, y_tr, X_te, y_te, X_val=make_real_data(data, seed, N_TR)
    y_tr_r=np.random.normal(0,np.std(y_tr),y_tr.shape)
    n_tr,p=X_tr.shape
    n_val=X_val.shape[0]
    n_te=X_te.shape[0]
    
    model_state = init_model(p, DIM_H, 1, dt, gamma)
    
    S_tr=jnp.zeros((n_tr,n_tr))
    S_val=jnp.zeros((n_val,n_tr))
    S_te=jnp.zeros((n_te,n_tr))
    
    S_tr_old=jnp.copy(S_tr)
    S_val_old=jnp.copy(S_val)
    S_te_old=jnp.copy(S_te)
    
    for epoch in range(301):
      if epoch % 5==0:
        r2_trs_seed.append(r2(y_tr, S_tr@y_tr))
        r2_tes_seed.append(r2(y_te, S_te@y_tr))
        epochs_seed.append(epoch)
        for alg in ALGS:
          errors_seed[alg].append(eval(alg)(S_tr,S_val,y_tr,y_tr_r))
        if r2(y_tr, S_tr@y_tr)<-1000: #things diverge
          break
        if r2(y_tr, S_tr@y_tr)>0.999:
          break
      
      if epoch % 20 == 0:
        K_tr, K_val, K_te=get_K(X_tr,X_val,X_te,model_state)
      model_state = train_step(model_state, X_tr, y_tr_r)
      S_tr, S_tr_old, S_val, S_val_old, S_te, S_te_old = get_S(K_tr,K_val,K_te,S_tr, S_val,S_te,S_tr_old,S_val_old,S_te_old,dt,gamma)
     
    for alg in ALGS:
      r2_trs[alg].append(r2_trs_seed[np.argmin(errors_seed[alg])])
      r2_tes[alg].append(r2_tes_seed[np.argmin(errors_seed[alg])])
      stop_times[alg].append(epochs_seed[np.argmin(errors_seed[alg])])
  for ii, alg in enumerate(ALGS):
    if ii==0:
      print(('\\multirow{'+str(len(ALGS))+'}{*}{'+data+'}').ljust(22),end='')
    else:
      print(''.ljust(22),end='')
    print('& '+alg.ljust(9),end='')
    print(f'& {np.median(r2_tes[alg]):.2f} ({np.quantile(r2_tes[alg],0.25):.2f}, {np.quantile(r2_tes[alg],0.75):.2f}) '.ljust(23), end='')
    print(f'& {np.median(stop_times[alg]):.0f} ({np.quantile(stop_times[alg],0.25):.0f}, {np.quantile(stop_times[alg],0.75):.0f})'.ljust(20)+'\\\\')
  print('\\hline')
print('')
