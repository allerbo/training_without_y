import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D



def opt_risk(snr,gamma):
  return 1/2*(snr-snr/gamma-1+np.sqrt(4*snr+(1-snr+snr/gamma)**2))

def risk0_01(snr,gamma):
  return gamma/(1-gamma)

def risk0_1inf(snr,gamma):
  return snr*(1-1/gamma)+1/(gamma-1)

def risk0(snr,gamma):
  return risk0_01(snr,gamma)*(gamma<=1)+risk0_1inf(snr,gamma)*(gamma>1)

def risk_ssmm_122(snr,gamma):
  return 1+snr*(1-np.sqrt(2*gamma))**2/gamma

def risk_ssmm(snr,gamma):
  return risk0_01(snr,gamma)*(gamma<=0.5)+risk_ssmm_122(snr,gamma)*((gamma>0.5)&(gamma<2))+risk0_1inf(snr,gamma)*(gamma>=2)


fig, ax=plt.subplots(1,2,figsize=(8,3),gridspec_kw={'width_ratios': [2.5,1]})

gammas=np.geomspace(1e-1,1e1,1000)
for c, snr in enumerate([1,2,5,10]):
  _=ax[0].plot(gammas, risk_ssmm(snr,gammas),'C'+str(c))
  _=ax[0].plot(gammas, opt_risk(snr,gammas),'C'+str(c)+'--')
  _=ax[0].plot(gammas, risk0(snr,gammas),'C'+str(c)+':')


_=ax[0].set_xscale('log')
_=ax[0].set_xlabel('$\\gamma$')
_=ax[0].set_xticks([0.1,1,10])
_=ax[0].set_xticklabels([0.1,1,10])

_=ax[0].set_ylim([-.5,10.2])
_=ax[0].set_ylabel('$\\overline{R_X}(\\lambda,\\gamma,\\text{SNR})/\\sigma_\\varepsilon^2$')
_=ax[0].set_yticks([0,5,10])


gammas2=np.linspace(0.5,2,100000)

snrs=np.linspace(1,80,1000)
max_qs=[]
for snr in snrs:
  max_qs.append(np.max(risk_ssmm_122(snr,gammas2)/opt_risk(snr,gammas2)))

#print(max(max_qs))

_=ax[1].plot(snrs,np.array(max_qs),'C4')
_=ax[1].set_xlabel('SNR')
_=ax[1].set_xticks([0,40,80])

_=ax[1].set_ylabel('$\\max_\\gamma \\ \\overline{R_X}(\\overline{\\lambda_T},\\gamma,\\text{SNR})/\\overline{R_X}(\\overline{\\lambda^*},\\gamma,\\text{SNR})$')
_=ax[1].set_yticks([1,2])
_=ax[1].set_ylim([0.9,2.6])

lines=[Line2D([0],[0],color='C0',lw=2),Line2D([0],[0],color='C1',lw=2),Line2D([0],[0],color='C2',lw=2),Line2D([0],[0],color='C3',lw=2)]

labs = ['SNR=1', 'SNR=2', 'SNR=5', 'SNR=10']
fig.tight_layout()
fig.subplots_adjust(bottom=.25)
fig.legend(lines, labs, loc='lower center', ncol=4)
plt.savefig('figures/mar_pas.pdf')
