import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import mode


def kern_class(K, y):
  y = np.asarray(y).ravel()
  labels = np.unique(y)
  # indicator matrix: shape (c, num_labels)
  M = (y[:, None] == labels[None, :]).astype(int)
  # weighted counts per row and label
  counts = K @ M
  return (labels[counts.argmax(axis=1)]).reshape(-1,1)

def kern_reg(K,y):
  return K@y/(K@np.ones(y.shape))

#kNN
def get_K_knn(k, X_tr, X_vt, y, reg):
  knn=KNeighborsRegressor(n_neighbors=k) if reg else KNeighborsClassifier(n_neighbors=k)
  _=knn.fit(X_tr, np.squeeze(y))
  n_tr=X_tr.shape[0]
  n_vt=X_vt.shape[0]
  K_tr=np.zeros((n_tr,n_tr), dtype=int)
  K_tr[np.arange(n_tr)[:, None], knn.kneighbors(X_tr)[1]]=1
  K_vt=np.zeros((n_vt,n_tr), dtype=int)
  K_vt[np.arange(n_vt)[:, None], knn.kneighbors(X_vt)[1]]=1
  return K_vt


def get_k_msv(X_tr, X_val, y, reg):
  n_tr=X_tr.shape[0]
  n_val=X_val.shape[0]
  best_msv=np.inf
  for k in (*range(1,min(n_tr-1,31)),n_tr):
    K_val = get_K_knn(k, X_tr, X_val, y, reg)
    S_val = K_val/(K_val@np.ones(y.shape))
    msv=np.linalg.norm(1/n_tr*np.eye(n_tr)-1/n_val*S_val.T@S_val)
    if msv<best_msv:
      best_msv=msv
      best_k=k
  return best_k

def get_k_gcv(X_tr, y, reg):
  n_tr=X_tr.shape[0]
  best_gcv=np.inf
  for k in (*range(2,min(n_tr-1,31)),n_tr):
    K_tr = get_K_knn(k, X_tr, X_tr, y, reg)
    S_tr = K_tr/(K_tr@np.ones(y.shape))
    IS=np.eye(n_tr)-S_tr
    gcv = np.linalg.norm(IS.T@IS)/(1e-8+np.trace(IS)**2)
    if gcv<best_gcv:
      best_gcv=gcv
      best_k=k
  return best_k


def get_k_loocv(X_tr, y, reg):
  n_tr=X_tr.shape[0]
  best_loocv=np.inf
  for k in (*range(2,min(n_tr-1,31)),n_tr):
    K_tr = get_K_knn(k, X_tr, X_tr, y, reg)
    S_tr = K_tr/(K_tr@np.ones(y.shape))
    IS=np.eye(n_tr)-S_tr
    loocv = np.linalg.norm(IS.T@np.diag(1/np.diag(1e-8+IS)**2)@IS)
    if loocv<best_loocv:
      best_loocv=loocv
      best_k=k
  return best_k

def get_k_cv(X_tr, y_tr, reg, folds=10):
  knn=KNeighborsRegressor() if reg else KNeighborsClassifier()
  n_tr=y_tr.shape[0]
  param_grid = {'n_neighbors': (*range(1, min(n_tr-1,31)),n_tr//10*9)}
  grid = GridSearchCV(knn, param_grid, cv=folds)
  grid.fit(X_tr, np.squeeze(y_tr))
  return grid.best_params_['n_neighbors']

def knn_predict(X_tr, X_val, X_te, y_tr, y_type):
  np.random.seed(0)
  reg = y_type.startswith('knnr')
  if y_type[-1]=='y':
    k=get_k_cv(X_tr, y_tr, reg)
    knn=KNeighborsRegressor(n_neighbors=k) if reg else KNeighborsClassifier(n_neighbors=k)
    _=knn.fit(X_tr, np.squeeze(y_tr))
    return knn.predict(X_te).reshape(-1,1), k
  y_tr_0=np.zeros(y_tr.shape) if reg else np.ones(y_tr.shape).astype(np.int32)
  k = get_k_msv(X_tr, X_val, y_tr_0, reg) if y_type[-1]=='s' else get_k_gcv(X_tr, y_tr_0, reg)
  K_te=get_K_knn(k, X_tr, X_te, y_tr_0, reg)
  fh_te = kern_reg(K_te,y_tr) if reg else kern_class(K_te,y_tr)
  return fh_te, k


#Random Forest

def get_Ks_rf(rf, X_tr, X_vt):
  P_vt_all=rf.decision_path(X_vt)[0].todense()
  P_tr_all=rf.decision_path(X_tr)[0].todense()
  starts_vt=rf.decision_path(X_vt)[1]
  starts_tr=rf.decision_path(X_tr)[1]
  n=X_tr.shape[0]
  assert np.all(starts_vt==starts_tr)
  Ks=[]
  for ii,es in zip(range(len(starts_vt)-1),rf.estimators_samples_):
    P_vt=P_vt_all[:,starts_vt[ii]:(starts_vt[ii+1])]
    P_tr=P_tr_all[:,starts_vt[ii]:(starts_vt[ii+1])]
    S=np.zeros((n,n))
    S[np.arange(n),es]=1
    K1=P_vt@P_tr.T@S.T
    K=np.asarray((K1==np.max(K1,0))@S).astype(np.int32)
    Ks.append(K)
  return Ks

def rf_predict(X_tr, X_val, X_te, y_tr, y_type):
  np.random.seed(0)
  reg = y_type.startswith('rfr')
  rf=RandomForestRegressor(random_state=0) if reg else RandomForestClassifier(random_state=0)
  if y_type[-1]=='y':
    #n_est=get_n_est_cv(X_tr, y_tr, reg)
    _=rf.fit(X_tr,np.squeeze(y_tr))
    return rf.predict(X_te).reshape(-1,1)
  y_tr_r=np.random.normal(0,1,y_tr.shape) if reg else np.random.randint(1,np.max(y_tr)+1,y_tr.shape).astype(np.int32)
  #y_tr_r=np.zeros(y_tr.shape) if reg else np.ones(y_tr.shape).astype(np.int32)
  _=rf.fit(X_tr,np.squeeze(y_tr_r))
  Ks_te=get_Ks_rf(rf, X_tr, X_te)
  fhs_te=[]
  for K_te in Ks_te:
    fhs_te.append(kern_reg(K_te,y_tr) if reg else kern_class(K_te,y_tr))
  fh_te = np.mean(fhs_te,0) if reg else ((mode(np.hstack(fhs_te),axis=1).mode).reshape(-1,1)) #Mean predictions across trees if reg, else #Maximum vote across all trees
  return fh_te

