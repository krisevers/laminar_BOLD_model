import numpy as np
import copy

import IPython

def neuronal_NVC_model(U, P):
	'''
	INPUT:
		K - Number of cortical depths

	OUTPUT:
		Y - structure with all baseline and relative physiological variables

	AUTHOR: Martin Havlicek, 5 August, 2019
	'''

	K = P['K']

	# Neuronal parameters:
	
	#--------------------------------------------------------------------------
	sigma   = P['sigma']     # self-inhibitory connection 
	mu      = P['mu']        # inhibitory-excitatory connection
	lambda_ = P['lambda']    # inhibitory gain
	Bsigma  = P['Bsigma']    # modulatory parameter of self-inhibitory connection
	Bmu     = P['Bmu']       # modulatory parameter of inhibitory-excitatory connection
	Blambda = P['Blambda']   # modulatory parameter of inhibitory connection
	C       = P['C']
	# NVC parameters:
	#--------------------------------------------------------------------------
	c1      = P['c1']
	c2      = P['c2']
	c3      = P['c3']

	# Initial condtions:
	Xn  = np.zeros((K,4))
	yn  = np.zeros((K,4))

	dt  = P['dt']
	neuro = np.zeros((int(P['T']/dt),K))
	cbf   = np.zeros((int(P['T']/dt),K))
	for t in range(int(P['T']/dt)):
	    Xn[:,3] = np.exp(Xn[:,3])

	    A = np.eye(K)*sigma
	    MU = np.ones(K)*mu
	    LAM = np.ones(K)*lambda_
	    for i in range(len(Bsigma)):
	        A = A + diag(Bsigma[:,i])*U['m'][t,i]
	    for i in range(len(Bmu)):
	        MU = MU + Bmu[:,i]*U['m'][t,i]
	    for i in range(len(Blambda)):
	        LAM = LAM + Blambda[:,i]*U['m'][t,i]

	    #----------------------------------------------------------------------
	    # Neuronal (excitatory & inhibitory)
	    yn[:,0]   = yn[:,0] + dt*(np.dot(A, Xn[:,0]) - MU*Xn[:,1] + np.dot(C, U['u'][t,:].T))

	    yn[:,1]   = yn[:,1] + dt*(LAM*(-Xn[:,1] + Xn[:,0]))
	    #----------------------------------------------------------------------
	    # Vasoactive signal:
	    yn[:,2]   = yn[:,2] + dt*(Xn[:,0] - c1*(Xn[:,2]))
	    #----------------------------------------------------------------------
	    # Inflow:
	    df_a      = c2*Xn[:,2] - c3*(Xn[:,3]-1)
	    yn[:,3]   = yn[:,3] + dt*(df_a/Xn[:,3])
	    
	    Xn         = copy.deepcopy(yn)

	    cbf[t,:]   = np.exp(yn[:,3]).T
	    neuro[t,:] = yn[:,0].T

	return neuro, cbf

def neuronal_NVC_parameters(K, P):
	'''
	INPUT:
		K - Numer of cortical depths
	
	OUTPUT:
		P - structure with all default parameters for neuronal-NVC model

	AUTHOR: Matrin Havlicek, 5 August, 2019
	'''


	P['K'] = K

	if K<10:
	    P['dt'] = 0.01 	# default integration step
	elif K<20:
	    P['dt'] = 0.005   # smaller for higher number of cortical depths
	else:
	    P['dt'] = 0.001

	# Neuronal parameter:
	#--------------------------------------------------------------------------
	P['sigma']   = -3
	P['mu']      = 1.5
	P['lambda']  = 0.2
	P['Bsigma']  = []
	P['Bmu']     = []
	P['Blambda'] = []
	P['C']       = np.eye(K)

	# NVC parameters:
	# --------------------------------------------------------------------------
	P['c1']      = 0.6
	P['c2']      = 1.5
	P['c3']      = 0.6

	return P