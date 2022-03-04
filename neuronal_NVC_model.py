import numpy as np

def neuronal_NVC_model(P, U):
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
	yn  = Xn

	dt  = P['dt']
	neuro = np.zeros(int(P['T']/dt),K)
	cbf   = np.zeros(int(P['T']/dt),K)
	for t in range(int(P['T']/dt)):
	    Xn[:,3] = np.exp(Xn[:,3])

	    A = np.eye(K)*sigma
	    MU = np.ones(K)*mu
	    LAM = np.ones(K)*lambda_
	    for i in range(np.shape(Bsigma,axis=1)):
	        A = A + diag(Bsigma[:,i])*U['m'][t,i]
	    for i in range(np.shape(Bmu,axis=1)):
	        MU = MU + Bmu[:,i]*U['m'][t,i]
	    for i in range(np.shape(Blambda, axis=1)):
	        LAM = LAM + Blambda[:,i]*U['m'][t,i]
	    
	    #----------------------------------------------------------------------
	    # Neuronal (excitatory & inhibitory)
	    yn[:,0]   = yn[:,0] + dt*(A*Xn[:,0] - MU*Xn[:,1] + C*U['u'][t,:].T)

	    yn[:,1]   = yn[:,1] + dt*(LAM*(-Xn[:,1] +  Xn[:,0]))
	    #----------------------------------------------------------------------
	    # Vasoactive signal:
	    yn[:,2]   = yn[:,2] + dt*(Xn[:,1] - c1*(Xn[:,3]))
	    #----------------------------------------------------------------------
	    # Inflow:
	    df_a      = c2*Xn[:,2] - c3*(Xn[:,3]-1)
	    yn[:,3]   = yn[:,3] + dt*(df_a/Xn[:,3])
	    
	    Xn         = yn
	     
	    cbf[t,:]   = np.exp(yn[:,3]).T
	    neuro[t,:] = yn[:,0].T


	return neuro, cbf