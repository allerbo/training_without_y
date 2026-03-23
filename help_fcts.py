import numpy as np
import pandas as pd
import pickle

from knn_rf_fcts import knn_predict, rf_predict
from lrr_krr_fcts import lrr_predict, krr_predict, spline_predict
from nnr_fcts import nnr_predict
from nnc_fcts import nnc_predict

def make_real_data(data, seed, n_tr=500, n_te=100):
  if data=='super':
    dm_all=pd.read_csv('csv_files/super.csv',sep=',').to_numpy()
    dm_all=np.roll(dm_all,1,1)
  elif data=='cpu':
    dm_all=pd.read_csv('csv_files/compactiv.csv',sep=',').to_numpy()
    dm_all=np.roll(dm_all,1,1)
  elif data=='power':
    dm_all=pd.read_csv('csv_files/power.csv',sep=',').iloc[:,1:].to_numpy()
    dm_all=np.roll(dm_all,1,1)
  elif data=='steel':
    dm_all=pd.read_csv('csv_files/steel.csv',sep=',').to_numpy()
  elif data=='cifar':
    with open('csv_files/cifar-10-batches-py/data_batch_1','rb') as cf:
      cd = pickle.load(cf,encoding='bytes')
    dm_all=np.hstack((np.array(cd[b'labels']).reshape(-1,1),cd[b'data']))
  elif data=='mnist':
    dm_all=pd.read_csv('csv_files/mnist_train.csv',sep=',').to_numpy().astype(float)
    dm_all[:,1:]+=np.random.normal(0,0.001,dm_all[:,1:].shape)
    
  np.random.seed(seed)
  np.random.shuffle(dm_all)
  p=dm_all.shape[1]-1
  n_val=n_tr
  dm=dm_all[:(n_tr+n_te),:]
  X=dm[:,1:]
  X=(X-np.mean(X, 0))/np.std(X,0)
  y=dm[:,0].reshape((-1,1))
  if not data in ['cifar','mnist']:
    y=y-np.mean(y)
  
  X_tr=X[:n_tr,:]
  X_te=X[n_tr:,:]
  y_tr=y[:n_tr,:]
  y_te=y[n_tr:,:]
  
  X_val=np.random.multivariate_normal(np.mean(X_tr,0),np.cov(X_tr.T),n_val)
  
  return X_tr, y_tr, X_te, y_te, X_val

def predict(X_tr, X_val, X_te, y_tr, alg, DIM_H=200, dt=1e-4, gamma=0.7, max_epoch=200):
  if alg.startswith('knn'):
    fh_te, k=knn_predict(X_tr, X_val, X_te, y_tr, alg)
  elif alg.startswith('rf'):
    fh_te=rf_predict(X_tr, X_val, X_te, y_tr, alg)
  elif alg.startswith('lrr'):
    fh_te=lrr_predict(X_tr, X_val, X_te, y_tr, alg)
  elif alg.startswith('krr'):
    fh_te=krr_predict(X_tr, X_val, X_te, y_tr, alg)
  elif alg.startswith('spl'):
    fh_te=spline_predict(X_tr, X_val, X_te, y_tr, alg)
  elif alg.startswith('nnc'):
    fh_te=nnc_predict(X_tr, X_val, X_te, y_tr, alg, DIM_H, dt, gamma, max_epoch)
  elif alg.startswith('nnr'):
    if max_epoch<10000:
      max_epoch=1000 if alg=='nnr_y' else 300 #Ugly fix to accomodate both real_data and wo_y_demo
    fh_te=nnr_predict(X_tr, X_val, X_te, y_tr, alg, DIM_H, dt, gamma, max_epoch)
  return fh_te


def acc(y,fh):
  return np.sum(y==fh)/y.shape[0]

def r2(y,fh):
  return 1-np.mean((y-fh)**2)/np.mean((y-np.mean(y))**2)

def print_tab(data, algs, accr2_tes):
  for ii, alg in enumerate(algs):
    if ii==0:
      print(('\\multirow{'+str(len(algs))+'}{*}{'+data+'}').ljust(22),end='')
    else:
      print(' &'.ljust(22),end='')
    print('& '+alg.ljust(9),end='')
    print(f'& {np.median(accr2_tes[alg]):.2f} ({np.quantile(accr2_tes[alg],0.25):.2f}, {np.quantile(accr2_tes[alg],0.75):.2f}) '.ljust(23), end='')
    print('\\\\')
  print('\\hline')

