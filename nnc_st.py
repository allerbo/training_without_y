import numpy as np
from matplotlib import pyplot as plt
import time
from help_fcts import acc, make_real_data
from nnc_fcts import init_model, train_step
import os, sys
os.environ['JAX_ENABLE_X64']='True'





dt=1e-4
gamma=0.7
N_TR=500
DIM_H=200

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

for data in ['mnist','cifar']:
  acc_trs=[]
  acc_tes=[]
  stop_times=[]
  
  for seed in range(10):
    acc_trs_seed=[]
    acc_vals_seed=[]
    acc_tes_seed=[]
    epochs_seed=[]
    
    X_tr_val, y_tr_val, X_te, y_te, _=make_real_data(data, seed, N_TR)
    n_tr_val=X_tr_val.shape[0]
    X_tr=X_tr_val[:int(0.8*n_tr_val),:]
    X_val=X_tr_val[int(0.8*n_tr_val):,:]
    y_tr=y_tr_val[:int(0.8*n_tr_val)]
    y_val=y_tr_val[int(0.8*n_tr_val):]
    n_tr,p=X_tr.shape
    n_val=X_val.shape[0]
    n_te=X_te.shape[0]
    dim_y=y_tr.shape[1]
    
    
    model_state = init_model(p,DIM_H,dim_y, dt, gamma)
    
    
    for epoch in range(201):
      if epoch % 5==0:
        acc_trs_seed.append(acc(y_tr,model_state.apply_fn(model_state.params,X_tr)[:,:-1]))
        acc_vals_seed.append(acc(y_val,model_state.apply_fn(model_state.params,X_val)[:,:-1]))
        acc_tes_seed.append(acc(y_te,model_state.apply_fn(model_state.params,X_te)[:,:-1]))
        epochs_seed.append(epoch)
      model_state = train_step(model_state, X_tr, y_tr)
    
    acc_trs.append(acc_trs_seed[np.argmax(acc_vals_seed)])
    acc_tes.append(acc_tes_seed[np.argmax(acc_vals_seed)])
    stop_times.append(epochs_seed[np.argmax(acc_vals_seed)])
  print(('\\multirow{x}{*}{'+data+'}').ljust(22),end='')
  print('& Standard'.ljust(9),end='')
  print(f'& {np.median(acc_tes):.2f} ({np.quantile(acc_tes,0.25):.2f}, {np.quantile(acc_tes,0.75):.2f}) '.ljust(23), end='')
  print(f'& {np.median(stop_times):.0f} ({np.quantile(stop_times,0.25):.0f}, {np.quantile(stop_times,0.75):.0f})'.ljust(20)+'\\\\')
  print('\\hline')
print('')
