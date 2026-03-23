from help_fcts import make_real_data, predict, acc, r2, print_tab
import sys


N_TR=500
DATA=['mnist', 'cifar', 'steel','cpu','super','power']
CLASS_ALGS=['nnc_y', 'nnc_s', 'nnc_g', 'knnc_y', 'knnc_s', 'knnc_g', 'rfc_y', 'rfc_r']
REG_ALGS=['lrr_y', 'lrr_s', 'lrr_g', 'krr_y', 'krr_s', 'krr_g', 'nnr_y', 'nnr_s', 'nnr_g', 'knnr_y', 'knnr_s', 'knnr_g', 'rfr_y', 'rfr_r']

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

seeds=range(10)
for data in DATA:
  if data in ['mnist', 'cifar']:
    algs=CLASS_ALGS
    accr2=acc
  else:
    algs=REG_ALGS
    accr2=r2
  
  accr2_tes={}
  for alg in algs:
    accr2_tes[alg]=[]
  for seed in seeds:
    X_tr, y_tr, X_te, y_te, X_val=make_real_data(data, seed, N_TR)
    for alg in algs:
      fh_te=predict(X_tr, X_val, X_te, y_tr, alg)
      accr2_tes[alg].append(accr2(y_te, fh_te))
  print_tab(data, algs, accr2_tes)

