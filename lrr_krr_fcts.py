import numpy as np
from scipy.interpolate import BSpline

def get_K_gauss(Xa,Xb, sigma):
  Xa2=np.sum(Xa**2,1).reshape((-1,1))
  XaXb=Xa.dot(Xb.T)
  Xb2=np.sum(Xb**2,1).reshape((-1,1))
  D2=Xa2-2*XaXb+Xb2.T
  return np.exp(-0.5*D2/sigma**2)

def make_spline(x_tr,x_vt,dt=0.001):
  x_tr=np.sort(x_tr,axis=0)
  x_vt=np.sort(x_vt,axis=0)
  t=np.hstack((np.repeat(x_tr[0]-dt,4),np.squeeze(x_tr),np.repeat(x_tr[-1]+dt,4)))
  bs=BSpline(t,np.eye(len(t)-4),3)
  B_tr=bs(np.squeeze(x_tr))
  B_vt=bs(np.squeeze(x_vt))
  B_vt2=bs.derivative(2)(np.squeeze(x_vt))
  if len(B_vt.shape)==1:
    B_vt=B_vt.reshape(1,-1)
  if len(B_vt2.shape)==1:
    B_vt2=B_vt2.reshape(1,-1)
  Omega=np.zeros((B_vt2.shape[1],B_vt2.shape[1]))
  for ti in range(B_vt2.shape[1]):
    for tj in range(ti+1):
      omij=dt*np.sum(B_vt2[:,ti]*B_vt2[:,tj])
      Omega[ti,tj]=omij
      Omega[tj,ti]=omij
  return B_tr,B_vt,Omega


def get_msv(X_tr, X_val, lbda, sigma=None, k_type='lin'):
  S_val = get_S(X_tr, X_val, lbda, sigma, k_type)
  n_val, n_tr=S_val.shape
  return np.linalg.norm(1/n_tr*np.eye(n_tr)-1/n_val*S_val.T@S_val)

def get_gcv(X_tr, X_val, lbda, sigma=None, k_type='lin'): #X_val not used
  S_tr = get_S(X_tr, X_tr, lbda, sigma, k_type)
  n_tr=S_tr.shape[0]
  IS=np.eye(n_tr)-S_tr
  return np.linalg.norm(IS.T@IS)/(1e-8+np.trace(IS)**2)

def get_loocv(X_tr, X_val, lbda, sigma=None, k_type='lin'): #X_val not used
  S_tr = get_S(X_tr, X_tr, lbda, sigma, k_type)
  n_tr=S_tr.shape[0]
  IS=np.eye(n_tr)-S_tr
  return np.linalg.norm(IS.T@np.diag(1/np.diag(1e-8+IS)**2)@IS)



def get_S(X_tr, X_vt, lbda, sigma=None, k_type='lin'):
  n_tr,p=X_tr.shape
  if k_type=='gauss':
    K_vt=get_K_gauss(X_vt, X_tr, sigma)
    K_tr=get_K_gauss(X_tr, X_tr, sigma)
    S=K_vt@np.linalg.inv(K_tr+lbda*np.eye(n_tr))
  elif k_type=='lin':
    if n_tr>p:
      S=X_vt@np.linalg.inv(X_tr.T@X_tr+lbda*np.eye(p))@X_tr.T
    else:
      S=X_vt@X_tr.T@np.linalg.inv(X_tr@X_tr.T+lbda*np.eye(n_tr))
  elif k_type=='spline':
    B_tr, B_vt, Omega=make_spline(X_tr,X_vt)
    try:
      S=B_vt@np.linalg.inv(B_tr.T@B_tr+lbda*Omega+1e-8*np.eye(B_tr.shape[1]))@B_tr.T
    except:
      S=np.zeros((B_vt.shape[0], B_tr.shape[0]))
  return S

def get_lbda_sigma(X_tr, X_val, lbdas, sigmas=[0], k_type='lin', msv_gcv='msv'):
  best_lbda=0
  best_sigma=0
  best_msv_gcv=np.inf
  get_msv_gcv = get_msv if msv_gcv=='msv' else get_gcv
  for sigma in sigmas:
    for lbda in lbdas:
      msv_gcv_val=get_msv_gcv(X_tr, X_val, lbda, sigma, k_type)
      if msv_gcv_val<best_msv_gcv:
        best_lbda=lbda
        best_sigma=sigma
        best_msv_gcv=msv_gcv_val
  return best_lbda, best_sigma

def cv10(X, y, lbdas, sigmas=[0], k_type='lin'):
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
        fh_val=get_S(X_tr, X_val, lbda, sigma, k_type)@y_tr
        mses.append(np.mean((y_val-fh_val)**2))
      if np.mean(mses)<best_mse:
        best_mse=np.mean(mses)
        best_lbda=lbda
        best_sigma=sigma
  return best_lbda, best_sigma


def lrr_predict(X_tr, X_val, X_te, y_tr, y_type,lbdas=np.hstack(([1e-6],np.geomspace(1e-3,100,10),[1e6]))):
  if y_type=='lrr_y':
    lbda, _ = cv10(X_tr, y_tr, lbdas, k_type='lin')
  elif y_type=='lrr_s':
    lbda, _ = get_lbda_sigma(X_tr, X_val, lbdas, k_type='lin', msv_gcv='msv')
  elif y_type=='lrr_g':
    lbda, _ = get_lbda_sigma(X_tr, X_val, lbdas, k_type='lin', msv_gcv='gcv')
  return get_S(X_tr, X_te, lbda, k_type='lin')@y_tr

def krr_predict(X_tr, X_val, X_te, y_tr, y_type, lbdas=np.hstack(([1e-6],np.geomspace(1e-3,100,10),[1e6])), sigmas=np.hstack((np.geomspace(1e-4,200,20),[1e6]))):
  if y_type=='krr_y':
    lbda, sigma = cv10(X_tr, y_tr, lbdas, sigmas, k_type='gauss')
  elif y_type=='krr_s':
    lbda, sigma = get_lbda_sigma(X_tr, X_val, lbdas, sigmas, k_type='gauss', msv_gcv='msv')
  elif y_type=='krr_g':
    lbda, sigma = get_lbda_sigma(X_tr, X_val, lbdas, sigmas, k_type='gauss', msv_gcv='gcv')
  return get_S(X_tr, X_te, lbda, sigma, k_type='gauss')@y_tr



def spline_predict(X_tr, X_val, X_te, y_tr, y_type,lbdas=np.hstack(([1e-10],np.geomspace(1e-4,1,100)))):
  if y_type=='spl_y':
    lbda, _ = cv10(X_tr, y_tr, lbdas, k_type='spline')
  elif y_type=='spl_s':
    lbda, _ = get_lbda_sigma(X_tr, X_val, lbdas, k_type='lin', msv_gcv='msv')
  elif y_type=='spl_g':
    lbda, _ = get_lbda_sigma(X_tr, X_val, lbdas, k_type='lin', msv_gcv='gcv')
  return get_S(X_tr, X_te, lbda, k_type='spline')@y_tr








