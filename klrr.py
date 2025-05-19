import numpy as np
from matplotlib import pyplot as plt
import time
from help_fcts import loogcv, cv10, ssmm, r2, make_real_data, kern
import sys


N_TR=500
ALGS=['cv10_y', 'ssmm_n', 'loocv_n']

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

def loocv_n(X_tr,X_val,y_tr,y_tr_r,lbdas,sigmas,nu):
  return loogcv(X_tr,lbdas,sigmas,y_tr=None,gcv=False,nu=nu)

def cv10_y(X_tr,X_val,y_tr,y_tr_r,lbdas,sigmas,nu):
  return cv10(X_tr,lbdas,sigmas,y_tr, nu=nu)

def ssmm_n(X_tr,X_val,y_tr,y_tr_r,lbdas,sigmas,nu):
  return ssmm(X_tr,X_val,lbdas,sigmas,y_tr=None,nu=nu)


lbdas_seed=np.hstack(([1e-6],np.geomspace(1e-3,100,10),[1e6]))
for NU in [0,100]:
  sigmas_seed=[0] if NU==0 else np.hstack((np.geomspace(1e-4,200,20),[1e6]))
  seeds=range(10)
  for data in ['steel','cpu','super','power']:
    r2_trs={}
    r2_tes={}
    lbdas={}
    sigmas={}
    for alg in ALGS:
      r2_trs[alg]=[]
      r2_tes[alg]=[]
      lbdas[alg]=[]
      sigmas[alg]=[]
  
    for seed in seeds:
      X_tr, y_tr, X_te, y_te, X_val=make_real_data(data, seed, N_TR)
      n_tr,p=X_tr.shape
      y_tr_r=np.random.normal(0,np.std(y_tr),y_tr.shape)
     
      for alg in ALGS:
        lbda,sigma=eval(alg)(X_tr,X_val,y_tr,y_tr_r,lbdas_seed, sigmas_seed, NU)
        if NU==0 and n_tr>p:
          fh_tr=X_tr@np.linalg.solve(X_tr.T@X_tr+lbda*np.eye(p),X_tr.T@y_tr)
          fh_te=X_te@np.linalg.solve(X_tr.T@X_tr+lbda*np.eye(p),X_tr.T@y_tr)
        else:
          K_tr=kern(X_tr,X_tr,sigma,NU)
          K_te=kern(X_te,X_tr,sigma,NU)
          fh_tr=K_tr@np.linalg.solve(K_tr+lbda*np.eye(n_tr),y_tr)
          fh_te=K_te@np.linalg.solve(K_tr+lbda*np.eye(n_tr),y_tr)
        r2_tes[alg].append(r2(y_te,fh_te))
        r2_trs[alg].append(r2(y_tr,fh_tr))
        lbdas[alg].append(lbda)
        sigmas[alg].append(sigma)
     
    for ii, alg in enumerate(ALGS):
      if ii==0:
        print(('\\multirow{'+str(len(ALGS))+'}{*}{'+data+'}').ljust(22),end='')
      else:
        print(''.ljust(22),end='')
      print('& '+alg.ljust(9),end='')
      print(f'& {np.median(r2_tes[alg]):.2f} ({np.quantile(r2_tes[alg],0.25):.2f}, {np.quantile(r2_tes[alg],0.75):.2f}) '.ljust(23), end='')
      print(f'& {np.median(lbdas[alg]):.2g} ({np.quantile(lbdas[alg],0.25):.2g}, {np.quantile(lbdas[alg],0.75):.2g})'.ljust(26), end='')
      if not NU==0:
        print(f'& {np.median(sigmas[alg]):.2g} ({np.quantile(sigmas[alg],0.25):.2g}, {np.quantile(sigmas[alg],0.75):.2g})'.ljust(24), end='')
      print('\\\\')
    print('\\hline')
  print('')
