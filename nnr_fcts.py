import jax
import numpy as np
import jax.numpy as jnp
from jax import jit
from flax import linen as nn
import optax
from flax.training.train_state import TrainState

def init_model(DIM_X, DIM_H, DIM_Y, dt, gamma, seed=0):
  rng, init_rng = jax.random.split(jax.random.PRNGKey(seed), 2)
  model=reg_fl(DIM_H, DIM_Y)
  theta=model.init(init_rng,jnp.ones((5,DIM_X)))
  opt=optax.sgd(dt, gamma)
  model_state = TrainState.create(apply_fn=model.apply, params=theta, tx=opt)
  return model_state

class reg_fl(nn.Module):
  DIM_H: int
  DIM_Y: int
  @nn.compact
  def __call__(self,x):
    x=nn.Dense(self.DIM_H,param_dtype=jnp.float64)(x)
    x=nn.activation.tanh(x)
    x=nn.Dense(self.DIM_Y,param_dtype=jnp.float64, kernel_init=nn.initializers.zeros)(x)
    return x


@jit
def train_step(model_state, x, y):
  def L2(theta):
    fh = model_state.apply_fn(theta, x)
    return 0.5*jnp.mean((fh-y)**2)
  
  loss, grads = jax.value_and_grad(L2)(model_state.params)
  model_state = model_state.apply_gradients(grads=grads)
  return model_state

@jit
def get_K(X_tr,X_val,X_te, model_state):
  def fh_th(X,theta,model_state):
    return model_state.apply_fn(theta,X)
  
  jac_dict_tr= jax.jacrev(fh_th,argnums=1)(X_tr,model_state.params,model_state)
  jac_dict_val= jax.jacrev(fh_th,argnums=1)(X_val,model_state.params,model_state)
  jac_dict_te= jax.jacrev(fh_th,argnums=1)(X_te,model_state.params,model_state)
  
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
      Ph_tr_s=jac_dict_tr['params'][k1][k2].reshape(n_tr,-1)
      Ph_val_s=jnp.squeeze(jac_dict_val['params'][k1][k2]).reshape(n_val,-1)
      Ph_te_s=jnp.squeeze(jac_dict_te['params'][k1][k2]).reshape(n_te,-1)
      K_tr+=Ph_tr_s@Ph_tr_s.T
      K_val+=Ph_val_s@Ph_tr_s.T
      K_te+=Ph_te_s@Ph_tr_s.T
  
  return K_tr, K_val, K_te

@jit
def get_S(K_tr,K_val,K_te,S_tr, S_val,S_te,S_tr_old,S_val_old,S_te_old,dt=0.001,gamma=0):
  IS=jnp.eye(S_tr.shape[0])-S_tr
  S_tr_new = S_tr +gamma*(S_tr-S_tr_old)  +dt*K_tr@IS
  S_val_new= S_val+gamma*(S_val-S_val_old)+dt*K_val@IS
  S_te_new = S_te +gamma*(S_te-S_te_old)  +dt*K_te@IS
  return S_tr_new, S_tr, S_val_new, S_val, S_te_new, S_te

