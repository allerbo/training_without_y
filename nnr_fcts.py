import numpy as np
import jax.numpy as jnp
import jax.random as jrand
from jax import jacrev, jit, value_and_grad
from flax import linen as nn
import optax
from flax.training.train_state import TrainState

def r2(y,fh):
  return 1-jnp.mean((y-fh)**2)/jnp.mean((y-jnp.mean(y))**2)

class reg_fl(nn.Module):
  DIM_H: int
  DIM_Y: int
  @nn.compact
  def __call__(self,x):
    x=nn.Dense(self.DIM_H,param_dtype=jnp.float64)(x)
    x=nn.activation.tanh(x)
    x=nn.Dense(self.DIM_Y,param_dtype=jnp.float64, kernel_init=nn.initializers.zeros)(x)
    return x

def init_model(DIM_X, DIM_H, DIM_Y, dt, gamma,seed=0):
  rng, init_rng = jrand.split(jrand.PRNGKey(seed), 2)
  model=reg_fl(DIM_H, DIM_Y)
  theta=model.init(init_rng,jnp.ones((1,DIM_X)))
  opt=optax.sgd(dt, gamma)
  model_state = TrainState.create(apply_fn=model.apply, params=theta, tx=opt)
  return model_state

@jit
def train_step(model_state, x, y):
  def L2(theta):
    fh = model_state.apply_fn(theta, x)
    return 0.5*jnp.mean((fh-y)**2)
  
  loss, grads = value_and_grad(L2)(model_state.params)
  model_state = model_state.apply_gradients(grads=grads)
  return model_state


@jit
def get_K(X_tr,X_val,X_te, model_state):
  def fh_th(X,theta,model_state):
    return model_state.apply_fn(theta,X)
  
  jac_dict_tr= jacrev(fh_th,argnums=1)(X_tr,model_state.params,model_state)
  jac_dict_val= jacrev(fh_th,argnums=1)(X_val,model_state.params,model_state)
  jac_dict_te= jacrev(fh_th,argnums=1)(X_te,model_state.params,model_state)
  
  k1=list(jac_dict_tr['params'].keys())[0]
  k2=list(jac_dict_tr['params'][k1].keys())[0]
  n_tr=jac_dict_tr['params'][k1][k2].shape[0]
  n_val=jac_dict_val['params'][k1][k2].shape[0]
  n_te=jac_dict_te['params'][k1][k2].shape[0]
  
  K_tr=jnp.zeros((n_tr,n_tr))
  K_val=jnp.zeros((n_val,n_tr))
  K_te=jnp.zeros((n_te,n_tr))
  
  for k1 in jac_dict_tr['params'].keys():
    for k2 in jac_dict_tr['params'][k1].keys():
      Ph_tr_s=jac_dict_tr['params'][k1][k2].reshape(n_tr,-1) #why no squeeze here?
      Ph_val_s=jnp.squeeze(jac_dict_val['params'][k1][k2]).reshape(n_val,-1)
      Ph_te_s=jnp.squeeze(jac_dict_te['params'][k1][k2]).reshape(n_te,-1)
      K_tr+=Ph_tr_s@Ph_tr_s.T
      K_val+=Ph_val_s@Ph_tr_s.T
      K_te+=Ph_te_s@Ph_tr_s.T
  
  return K_tr, K_val, K_te

@jit
def get_S(K_tr,K_val,K_te,S_tr, S_val,S_te,S_tr_old,S_val_old,S_te_old,dt,gamma):
  IS=jnp.eye(S_tr.shape[0])-S_tr
  S_tr_new = S_tr +gamma*(S_tr-S_tr_old)  +dt*K_tr@IS
  S_val_new= S_val+gamma*(S_val-S_val_old)+dt*K_val@IS
  S_te_new = S_te +gamma*(S_te-S_te_old)  +dt*K_te@IS
  return S_tr_new, S_tr, S_val_new, S_val, S_te_new, S_te

@jit
def get_msv(S_val):
  n_val,n_tr=S_val.shape
  I_n=jnp.eye(n_tr)
  return jnp.linalg.norm(1/n_tr*I_n-1/n_val*S_val.T@S_val)

@jit
def get_gcv(S_tr):
  n_tr=S_tr.shape[0]
  IS=jnp.eye(n_tr)-S_tr
  return jnp.linalg.norm(IS.T@IS)/(1e-8+jnp.trace(IS)**2)

@jit
def get_loocv(S_tr):
  n_tr=S_tr.shape[0]
  IS=jnp.eye(n_tr)-S_tr
  return jnp.linalg.norm(IS.T@jnp.diag(1/jnp.diag(1e-8+IS)**2)@IS)


def nnr_predict(X_tr, X_val, X_te, y_tr, y_type, DIM_H, dt, gamma, max_epoch=200, kern_epoch=20, save_epoch=5):
  np.random.seed(0)
  if y_type=='nnr_y':
    n_tr_val=X_tr.shape[0]
    X_val=X_tr[int(0.8*n_tr_val):,:]
    y_val=y_tr[int(0.8*n_tr_val):]
    X_tr=X_tr[:int(0.8*n_tr_val),:]
    y_tr=y_tr[:int(0.8*n_tr_val)]
    best_r2_val=-np.inf
    
  else:
    msv_gcv = 'msv' if y_type[-1]=='s' else 'gcv'
    y_tr_r=np.random.normal(0,1,y_tr.shape)
    #y_tr_r=np.zeros(y_tr.shape)
    
    n_tr=X_tr.shape[0]
    n_val=X_val.shape[0]
    n_te=X_te.shape[0]
    
    S_tr=jnp.zeros((n_tr,n_tr))
    S_val=jnp.zeros((n_val,n_tr))
    S_te=jnp.zeros((n_te,n_tr))
    
    S_tr_old=jnp.copy(S_tr)
    S_val_old=jnp.copy(S_val)
    S_te_old=jnp.copy(S_te)
    
    best_msv_gcv_val=np.inf
  
  fh_nnr=None
  model_state = init_model(X_tr.shape[1], DIM_H, 1, dt, gamma)
  
  if y_type=='nnr_y':
    for epoch in range(max_epoch+1):
      if epoch % save_epoch == 0:
        r2_tr=r2(y_tr,model_state.apply_fn(model_state.params,X_tr))
        if r2_tr<-1000 or r2_tr>0.999: #diverge or converge
          break
        r2_val=r2(y_val,model_state.apply_fn(model_state.params,X_val))
        if r2_val>best_r2_val:
          best_r2_val=r2_val
          fh_nnr=model_state.apply_fn(model_state.params,X_te)
      model_state = train_step(model_state, X_tr, y_tr)
  else:
    for epoch in range(max_epoch+1):
      if epoch % save_epoch == 0:
        r2_tr=r2(y_tr,S_tr@y_tr)
        if r2_tr<-1000 or r2_tr>0.999: #diverge or converge
          break
        msv_gcv_val=get_msv(S_val) if msv_gcv=='msv' else get_gcv(S_tr)
        if msv_gcv_val<best_msv_gcv_val:
          best_msv_gcv_val=msv_gcv_val
          fh_nnr=S_te@y_tr
      if epoch % kern_epoch==0:
        K_tr, K_val, K_te=get_K(X_tr,X_val,X_te,model_state)
      model_state = train_step(model_state, X_tr, y_tr_r)
      S_tr, S_tr_old, S_val, S_val_old, S_te, S_te_old = get_S(K_tr,K_val,K_te,S_tr, S_val,S_te,S_tr_old,S_val_old,S_te_old,dt,gamma)
  
  return fh_nnr
