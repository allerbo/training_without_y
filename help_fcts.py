import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
import pickle

def ssmm_S(S_val,y_tr=None):
  n_val,n_tr=S_val.shape
  I_n=np.eye(n_tr)
  if y_tr is None:
    return np.linalg.norm(1/n_tr*I_n-1/n_val*S_val.T@S_val)
  else:
    return np.abs(y_tr.T@(1/n_tr*I_n-1/n_val*S_val.T@S_val)@y_tr)[0][0]

def loogcv_S(S_tr,y_tr=None,gcv=False):
  n_tr=S_tr.shape[0]
  I_n=np.eye(n_tr)
  IS=I_n-S_tr
  if y_tr is None:
    if gcv:
      return np.linalg.norm(IS.T@IS)/(1e-8+np.trace(IS)**2)
    else:
      return np.linalg.norm(IS.T@np.diag(1/np.diag(1e-8+IS)**2)@IS)
  else:
    if gcv:
      return (y_tr.T@IS.T@IS@y_tr/(1e-8+np.trace(IS)**2))[0][0]
    else:
      return (y_tr.T@IS.T@np.diag(1/np.diag(1e-8+IS)**2)@IS@y_tr)[0][0]


def kern(Xa,Xb,sigma, nu=np.inf):
  if nu==0:
    return Xa@Xb.T
  Xa2=np.sum(Xa**2,1).reshape((-1,1))
  XaXb=Xa.dot(Xb.T)
  Xb2=np.sum(Xb**2,1).reshape((-1,1))
  D2=Xa2-2*XaXb+Xb2.T
  return np.exp(-0.5*D2/sigma**2)


def loogcv(X_tr,lbdas,sigmas, y_tr=None, gcv=False,nu=100):
  n_tr,p=X_tr.shape
  I_n=np.eye(n_tr)
  best_mse=np.inf
  best_lbda=0
  best_sigma=0
  for sigma in sigmas:
    K_tr=kern(X_tr,X_tr,sigma,nu)
    for lbda in lbdas:
      if nu==0 and n_tr>p:
        IS=I_n-X_tr@np.linalg.inv(X_tr.T@X_tr+lbda*np.eye(p))@X_tr.T
      else:
        IS=np.linalg.inv(K_tr+lbda*I_n)
      if y_tr is None:
        if gcv:
          mse=np.linalg.norm(IS.T@IS)/(np.trace(IS)**2)
        else:
          mse=np.linalg.norm(IS.T@np.diag(1/np.diag(IS)**2)@IS)
      elif np.linalg.norm(y_tr)==0:
        if gcv:
          mse=np.trace(IS.T@IS)/(np.trace(IS)**2)
        else:
          mse=np.trace(IS.T@np.diag(1/np.diag(IS)**2)@IS)
      else:
        if gcv:
          mse=y_tr.T@IS.T@IS@y_tr/(np.trace(IS)**2)
        else:
          mse=y_tr.T@IS.T@np.diag(1/np.diag(IS)**2)@IS@y_tr
      if mse<=best_mse:
        best_mse=mse
        best_lbda=lbda
        best_sigma=sigma
  return best_lbda, best_sigma


def ssmm(X_tr,X_val,lbdas,sigmas,y_tr=None, nu=100):
  n_tr,p=X_tr.shape
  n_val=X_val.shape[0]
  I_n=np.eye(n_tr)
  best_mse=np.inf
  best_lbda=0
  best_sigma=0
  for sigma in sigmas:
    K_tr=kern(X_tr,X_tr,sigma,nu)
    K_val=kern(X_val,X_tr,sigma,nu)
    for lbda in lbdas:
      if nu==0 and n_tr>p:
        A=np.linalg.inv(X_tr.T@X_tr+lbda*np.eye(p))@X_tr.T
        K_val=X_val
      else:
        A=np.linalg.inv(K_tr+lbda*I_n)
      if y_tr is None:
        mse=np.linalg.norm(1/n_val*A.T@K_val.T@K_val@A-1/n_tr*I_n)
      elif np.linalg.norm(y_tr)==0:
        mse=np.abs(np.trace(1/n_val*A.T@K_val.T@K_val@A-1/n_tr*I_n))
      else:
        mse=(y_tr.T@(1/n_val*A.T@K_val.T@K_val@A-1/n_tr*I_n)@y_tr)**2
      if mse<=best_mse:
        best_mse=mse
        best_lbda=lbda
        best_sigma=sigma
  return best_lbda, best_sigma

def cv10(X,lbdas,sigmas,y,nu=100):
  n=y.shape[0]
  np.random.seed(0)
  per=np.random.permutation(n)
  folds=np.array_split(per,10)
  best_mse=np.inf
  best_lbda=0
  best_sigma=0
  for sigma in sigmas:
    for lbda in lbdas:
      mses=[]
      for v_fold in range(len(folds)):
        t_folds=np.concatenate([folds[t_fold] for t_fold in range(len(folds)) if v_fold != t_fold])
        v_folds=folds[v_fold]
        X_tr=X[t_folds,:]
        y_tr=y[t_folds,:]
        X_val=X[v_folds,:]
        y_val=y[v_folds,:]
        K_val=kern(X_val, X_tr, sigma,nu)
        K_tr=kern(X_tr, X_tr, sigma,nu)
        fh_val=K_val@np.linalg.solve(K_tr+lbda*np.eye(K_tr.shape[0]),y_tr)
        mses.append(np.mean((y_val-fh_val)**2))
      if np.mean(mses)<best_mse:
        best_mse=np.mean(mses)
        best_lbda=lbda
        best_sigma=sigma
  return best_lbda, best_sigma

def r2(y,fh):
  return 1-np.mean((y-fh)**2)/np.mean((y-np.mean(y))**2)

def acc(y,fh):
  if not fh.shape[1]==y.shape[1]:
    fh=np.hstack((fh,1-np.sum(fh,1).reshape(-1,1)))
  return np.mean((np.argmax(fh,1)==np.argmax(y,1)))

def make_real_data(data, seed, n_tr=200):
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
    with open('csv_files/cifar_batch_1','rb') as cf:
      cd = pickle.load(cf,encoding='bytes')
    dm_all=np.hstack((np.array(cd[b'labels']).reshape(-1,1),cd[b'data']))
  elif data=='mnist':
    dm_all=pd.read_csv('csv_files/mnist_train.csv',sep=',').to_numpy().astype(float)
    dm_all[:,1:]+=np.random.normal(0,0.001,dm_all[:,1:].shape)
    
  np.random.seed(seed)
  np.random.shuffle(dm_all)
  p=dm_all.shape[1]-1
  n_te=100
  n_val=n_tr
  dm=dm_all[:(n_tr+n_te),:]
  
  X=dm[:,1:]
  
  X=(X-np.mean(X, 0))/np.std(X,0)
  
  y=dm[:,0].reshape((-1,1))
  if data in ['cifar','mnist']:
    n_class=len(np.unique(y))
    y=np.squeeze(np.eye(n_class)[y.astype(int)])
  else:
    y=y-np.mean(y)
  
  X_tr=X[:n_tr,:]
  X_te=X[n_tr:,:]
  y_tr=y[:n_tr,:]
  y_te=y[n_tr:,:]
  
  X_val=np.random.multivariate_normal(np.mean(X_tr,0),np.cov(X_tr.T),n_val)
  
  return X_tr, y_tr, X_te, y_te, X_val


def make_spline(x_tr,x_te,dt=0.001):
  t=np.hstack((np.repeat(x_tr[0]-dt,4),np.squeeze(x_tr),np.repeat(x_tr[-1]+dt,4)))
  bs=BSpline(t,np.eye(len(t)-4),3)
  B_tr=bs(np.squeeze(x_tr))
  B_te=bs(np.squeeze(x_te))
  B_te2=bs.derivative(2)(np.squeeze(x_te))
  Omega=np.zeros((B_te2.shape[1],B_te2.shape[1]))
  for ti in range(B_te2.shape[1]):
    for tj in range(ti+1):
      omij=dt*np.sum(B_te2[:,ti]*B_te2[:,tj])
      Omega[ti,tj]=omij
      Omega[tj,ti]=omij
  return B_tr,B_te,Omega

def ssmm_loocv(x_tr,x_te,lbdas, sigmas=[0], y_tr=None):
  b_krr=(len(sigmas)>1)
  n_tr=x_tr.shape[0]
  n_te=x_te.shape[0]
  I_n=np.eye(n_tr)
  best_mse=np.inf
  best_lbda=0
  best_sigma=0
  if not b_krr:
    B_tr,B_te,Omega=make_spline(x_tr,x_te)
  for sigma in sigmas:
    if b_krr:
      K_tr=kern(x_tr,x_tr,sigma,100)
      K_te=kern(x_te,x_tr,sigma,100)
    for lbda in lbdas:
      if y_tr is None:
        if b_krr:
          S=K_te@np.linalg.inv(K_tr+lbda*I_n)
        else:
          S=B_te@np.linalg.inv(B_tr.T@B_tr+lbda*Omega+1e-8*np.eye(B_tr.shape[1]))@B_tr.T
        mse=np.linalg.norm(1/n_te*S.T@S-1/n_tr*I_n)
      else:
        if b_krr:
          IS=I_n-K_tr@np.linalg.inv(K_tr+lbda*I_n)
        else:
          IS=I_n-B_tr@np.linalg.inv(B_tr.T@B_tr+lbda*Omega+1e-8*np.eye(B_tr.shape[1]))@B_tr.T
        mse=(y_tr.T@IS.T@np.diag(1/np.diag(1e-8+IS)**2)@IS@y_tr)[0][0]
      if mse<=best_mse:
        best_mse=mse
        best_lbda=lbda
        best_sigma=sigma
  return best_lbda, best_sigma

