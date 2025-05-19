import numpy as np
from matplotlib import pyplot as plt
import time
from help_fcts import r2, make_real_data
from nnr_fcts import init_model, train_step
import os, sys
os.environ['JAX_ENABLE_X64']='True'




DIM_H=200
dt=1e-4
gamma=0.7
N_TR=500

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

for data in ['steel','cpu','super','power']:
  r2_trs=[]
  r2_tes=[]
  stop_times=[]
  
  for seed in range(10):
    r2_trs_seed=[]
    r2_vals_seed=[]
    r2_tes_seed=[]
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

    model_state = init_model(p, DIM_H, 1, dt, gamma)
    
    for epoch in range(1001):
      model_state = train_step(model_state, X_tr, y_tr)
      
      if epoch % 5==0:
        r2_tr=r2(y_tr,model_state.apply_fn(model_state.params,X_tr))
        r2_val=r2(y_val,model_state.apply_fn(model_state.params,X_val))
        r2_te=r2(y_te,model_state.apply_fn(model_state.params,X_te))
        if r2_tr<-1000 or r2_tr>0.999:
          break
        r2_trs_seed.append(r2_tr)
        r2_vals_seed.append(r2_val)
        r2_tes_seed.append(r2_te)
        epochs_seed.append(epoch)
    
      r2_trs.append(r2_trs_seed[np.argmax(r2_vals_seed)])
      r2_tes.append(r2_tes_seed[np.argmax(r2_vals_seed)])
      stop_times.append(epochs_seed[np.argmax(r2_vals_seed)])

  print(('\\multirow{x}{*}{'+data+'}').ljust(22),end='')
  print('& Standard'.ljust(12),end='')
  print(f'& {np.median(r2_tes):.2f} ({np.quantile(r2_tes,0.25):.2f}, {np.quantile(r2_tes,0.75):.2f}) '.ljust(23), end='')
  print(f'& {np.median(stop_times):.0f} ({np.quantile(stop_times,0.25):.0f}, {np.quantile(stop_times,0.75):.0f})'.ljust(20)+'\\\\')
  print('\\hline')
print('')
