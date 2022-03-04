import numpy as np

def neuronal_NVC_parameters(K):
	'''
	INPUT:
		K - Numer of cortical depths
	
	OUTPUT:
		P - structure with all default parameters for neuronal-NVC model

	AUTHOR: Matrin Havlicek, 5 August, 2019
	'''
	K = P['K'] 

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
	P['C']       = eye(K)

	# NVC parameters:
	# --------------------------------------------------------------------------
	P['c1']      = 0.6
	P['c2']      = 1.5
	P['c3']      = 0.6

	return P