import numpy as np
import jax.numpy as jnp
import jax.random as jrand
from jax import jacrev, jit, value_and_grad
from flax import linen as nn
import optax
from flax.training.train_state import TrainState
import os
os.environ['JAX_ENABLE_X64']='True'
from jax import config as jc
jc.update('jax_platform_name', 'cpu') #quicker?

def acc(y,fh):
#  if not fh.shape[1]==y.shape[1]:
#    fh=jnp.hstack((fh,1-jnp.sum(fh,1).reshape(-1,1)))
  return jnp.mean((jnp.argmax(fh,1)==jnp.argmax(y,1)))

def init_model(DIM_X, DIM_H, DIM_Y, dt, gamma,seed=0):
  rng, init_rng = jrand.split(jrand.PRNGKey(seed), 2)
  model=class_fl(DIM_H, DIM_Y)
  theta=model.init(init_rng,jnp.ones([1,DIM_X]))
  opt=optax.sgd(dt, gamma)
  model_state = TrainState.create(apply_fn=model.apply, params=theta, tx=opt)
  return model_state

class class_fl(nn.Module):
  DIM_H: int
  DIM_Y: int
  @nn.compact
  def __call__(self,x):
    x=nn.Dense(self.DIM_H,param_dtype=jnp.float64)(x)
    x=nn.activation.relu(x)
    x=nn.Dense(self.DIM_Y,param_dtype=jnp.float64, kernel_init=nn.initializers.zeros)(x)
    x=nn.activation.softmax(x)
    return x

@jit
def train_step(model_state, x, y):
  def cross_entr(theta):
    fh = model_state.apply_fn(theta, x)
    return -(jnp.sum(jnp.log(fh+1e-7)*y))
  
  loss, grads = value_and_grad(cross_entr)(model_state.params)
  model_state = model_state.apply_gradients(grads=grads)
  return model_state

@jit
def get_K(X_tr,X_val,X_te, model_state):
  def fh_th(X,theta,model_state):
    return model_state.apply_fn(theta,X)[:,:-1]
  
  jac_dict_tr= jacrev(fh_th,argnums=1)(X_tr,model_state.params,model_state)
  jac_dict_val= jacrev(fh_th,argnums=1)(X_val,model_state.params,model_state)
  jac_dict_te= jacrev(fh_th,argnums=1)(X_te,model_state.params,model_state)
  
  k1=list(jac_dict_tr['params'].keys())[0]
  k2=list(jac_dict_tr['params'][k1].keys())[0]
  n_tr,dim_y=jac_dict_tr['params'][k1][k2].shape[:2]
  n_val=jac_dict_val['params'][k1][k2].shape[0]
  n_te=jac_dict_te['params'][k1][k2].shape[0]
  
  K_tr=jnp.zeros((n_tr,dim_y,n_tr,dim_y))
  K_val=jnp.zeros((n_val,dim_y,n_tr,dim_y))
  K_te=jnp.zeros((n_te,dim_y,n_tr,dim_y))
  
  for k1 in jac_dict_tr['params'].keys():
    for k2 in jac_dict_tr['params'][k1].keys():
      Ph_tr_s=jac_dict_tr['params'][k1][k2].reshape(n_tr,dim_y,-1)
      Ph_val_s=jac_dict_val['params'][k1][k2].reshape(n_val,dim_y,-1)
      Ph_te_s=jac_dict_te['params'][k1][k2].reshape(n_te,dim_y,-1)
      K_tr+=jnp.einsum('ijk,mnk',Ph_tr_s,Ph_tr_s)
      K_val+=jnp.einsum('ijk,mnk',Ph_val_s,Ph_tr_s)
      K_te+=jnp.einsum('ijk,mnk',Ph_te_s,Ph_tr_s)
  
  return K_tr, K_val, K_te

@jit
def get_S(K_tr,K_val,K_te,S_tr, S_val,S_te,S_tr_old,S_val_old,S_te_old,Ft,dt=0.001,gamma=0):
  IS=jnp.eye((S_tr.shape[0]*S_tr.shape[1])).reshape(S_tr.shape)-S_tr
  FtIS=jnp.einsum('ijk,iklm->ijlm',Ft,IS)
  S_tr_new= S_tr+gamma*(S_tr-S_tr_old)+dt*jnp.einsum('ijkl,klmn',K_tr,FtIS)
  S_val_new= S_val+gamma*(S_val-S_val_old)+dt*jnp.einsum('ijkl,klmn',K_val,FtIS)
  S_te_new= S_te+gamma*(S_te-S_te_old)+dt*jnp.einsum('ijkl,klmn',K_te,FtIS)
  return S_tr_new, S_tr, S_val_new, S_val, S_te_new, S_te

@jit
def get_Ft(X_tr,model_state):
  def fh2mat(fh):
    dim_y=fh.shape[0]
    return jnp.diag(1/fh)+1/(1-jnp.sum(fh)+1e-10)*jnp.ones((dim_y,dim_y))
  
  return jnp.apply_along_axis(fh2mat,1,model_state.apply_fn(model_state.params,X_tr)[:,:-1])


@jit
def get_msv(S_val):
  S_val=S_val.reshape(S_val.shape[0]*S_val.shape[1],S_val.shape[2]*S_val.shape[3])
  n_val,n_tr=S_val.shape
  I_n=jnp.eye(n_tr)
  return jnp.linalg.norm(1/n_tr*I_n-1/n_val*S_val.T@S_val)

@jit
def get_gcv(S_tr):
  S_tr=S_tr.reshape(S_tr.shape[0]*S_tr.shape[1],S_tr.shape[2]*S_tr.shape[3])
  n_tr=S_tr.shape[0]
  IS=jnp.eye(n_tr)-S_tr
  return jnp.linalg.norm(IS.T@IS)/(1e-8+jnp.trace(IS)**2)

@jit
def get_loocv(S_tr):
  S_tr=S_tr.reshape(S_tr.shape[0]*S_tr.shape[1],S_tr.shape[2]*S_tr.shape[3])
  n_tr=S_tr.shape[0]
  IS=jnp.eye(n_tr)-S_tr
  return jnp.linalg.norm(IS.T@jnp.diag(1/jnp.diag(1e-8+IS)**2)@IS)

def one_hot(y):
  n_class=len(np.unique(y))
  y=np.squeeze(np.eye(n_class)[y.astype(int)])
  return y

def nnc_predict(X_tr, X_val, X_te, y_tr, y_type, DIM_H, dt, gamma, max_epoch=200, kern_epoch=20, save_epoch=5):
  np.random.seed(0)
  y_tr=one_hot(y_tr)
  if y_type=='nnc_y':
    n_tr_val=X_tr.shape[0]
    X_val=X_tr[int(0.8*n_tr_val):,:]
    y_val=y_tr[int(0.8*n_tr_val):,:]
    X_tr=X_tr[:int(0.8*n_tr_val),:]
    y_tr=y_tr[:int(0.8*n_tr_val),:]
    best_acc_val=-np.inf
  else:
    n_tr=X_tr.shape[0]
    n_val=X_val.shape[0]
    n_te=X_te.shape[0]
    dim_y=y_tr.shape[1]
    msv_gcv = 'msv' if y_type[-1]=='s' else 'gcv'
    y_tr_r=np.eye(dim_y)[np.random.randint(0,dim_y,n_tr)]
    #y_tr_r=np.eye(dim_y)[np.zeros(n_tr).astype(int)]
    
    S_tr=jnp.zeros((n_tr,dim_y-1,n_tr,dim_y-1))
    S_val=jnp.zeros((n_val,dim_y-1,n_tr,dim_y-1))
    S_te=jnp.zeros((n_te,dim_y-1,n_tr,dim_y-1))
    
    S_tr_old=jnp.copy(S_tr)
    S_val_old=jnp.copy(S_val)
    S_te_old=jnp.copy(S_te)
    
    best_msv_gcv_val=np.inf
    
  model_state = init_model(X_tr.shape[1],DIM_H,y_tr.shape[1], dt, gamma)
  
  if y_type=='nnc_y':
    for epoch in range(max_epoch+1):
      if epoch % save_epoch == 0:
        acc_val=acc(y_val,model_state.apply_fn(model_state.params,X_val))
        if acc_val>best_acc_val:
          best_acc_val=acc_val
          fh_nnc=model_state.apply_fn(model_state.params,X_te)
      model_state = train_step(model_state, X_tr, y_tr)
    return np.argmax(fh_nnc, axis=1).reshape(-1,1)
  else:
    fh_tr0=model_state.apply_fn(model_state.params,X_tr)
    fh_val0=model_state.apply_fn(model_state.params,X_val)
    fh_te0=model_state.apply_fn(model_state.params,X_te)
    for epoch in range(max_epoch+1):
      if epoch % save_epoch == 0:
        msv_gcv_val=get_msv(S_val) if msv_gcv=='msv' else get_gcv(S_tr)
        if msv_gcv_val<best_msv_gcv_val:
          best_msv_gcv_val=msv_gcv_val
          fh_nnc=jnp.einsum('ijkl,kl',S_te,(y_tr-fh_tr0)[:,:-1])+fh_te0[:,:-1]
      if epoch % kern_epoch==0:
        K_tr, K_val, K_te=get_K(X_tr,X_val,X_te,model_state)
        Ft=get_Ft(X_tr,model_state)
      model_state = train_step(model_state, X_tr, y_tr_r)
      S_tr, S_tr_old, S_val, S_val_old, S_te, S_te_old = get_S(K_tr,K_val,K_te,S_tr, S_val,S_te,S_tr_old,S_val_old,S_te_old,Ft,dt,gamma)
      if jnp.isnan(jnp.sum(S_tr)):
        break
    
    fh_nnc=jnp.hstack((fh_nnc,1-jnp.sum(fh_nnc,1).reshape(-1,1)))
    return np.argmax(fh_nnc, axis=1).reshape(-1,1)
