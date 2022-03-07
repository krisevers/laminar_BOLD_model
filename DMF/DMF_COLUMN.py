import numpy as np
import pylab as plt

'''
Example Dynamic Mean Field (DMF) Column model with connectivity matrix from Potjans & Diesmann (2014)
'''

def H(x, g, Ithr, d):
	x.astype(float)
	return g*(x - Ithr) / (1 - np.exp(-d*g*(x - Ithr)))

def DMF(W, Wext, Iext, t_sim, params, stim, dt=1e-3):

	num_pops = np.shape(W)[0]
	num_circ = int(num_pops/2)

	# parameters
	g_E = params['g_E']
	g_I = params['g_I']
	g = np.tile([g_E, g_I], [num_circ])
	Ithr_E = params['Ithr_E']
	Ithr_I = params['Ithr_I']
	Ithr = np.tile([Ithr_E, Ithr_I], [num_circ])
	d_E = params['d_E']
	d_I = params['d_I']
	d = np.tile([d_E, d_I], [num_circ])

	tau_NMDA = .1
	tau_GABA = .01
	tau = np.tile([tau_NMDA, tau_GABA], [num_circ])

	# maybe external input is governed by AMPA 
	
	gamma_E = params['gamma_E']
	gamma_I = 1 
	gamma = np.tile([gamma_E, gamma_I], [num_circ])

	t_steps = int(t_sim/dt)

	sigma = 0.0
	dsig = np.sqrt(dt/tau_GABA) * sigma

	I = np.zeros((num_pops, t_steps))
	R = np.zeros((num_pops, t_steps))
	S = np.zeros((num_pops, t_steps))

	for t in range(1, t_steps):
		# add stimulus
		if t > stim['onset']/dt and t < stim['offset']/dt:
			Iext_ = Iext + stim['Iext']
		else:
			Iext_ = Iext
		I[:,t] = np.dot(W, S[:,t-1]) + Wext*Iext_
		R[:,t] = H(I[:,t], g, Ithr, d)
		D = 1 - S[:,t-1]
		D[1::2] = 1
		S[:,t] = S[:,t-1] + dt * (-S[:,t-1]/tau + D*gamma*R[:,t]) + dsig * np.random.randn(num_pops)
		S[S>1] = 1 
		S[S<0] = 0

	return I, R, S


if __name__=='__main__':

	num_pops = 8
	num_circ = 4

	params = {'Iext': 0.0513, 'Ithr_E': 0.566, 'Ithr_I': 0.737, 
			  'd_E': 0.438, 'd_I': 0.540, 'g_E': 1.592, 'g_I': 1.959, 
			  'gamma_E': 0.286}



	# connectivity
	def get_num_connections(P, N1, N2):
		return np.log(1-P) / np.log(1 - 1/(N1 * N2)) / N1

	g = -4
	J_E = 87.8e-3
	J_I = J_E * g
	J = np.tile([J_E, J_I], [num_pops, num_circ])
	P = np.array([[0.1009, 0.1689, 0.0837, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],   
	         	  [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],  
			   	  [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],  
			   	  [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],   
			  	  [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],   	
			  	  [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],   
			   	  [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],   	
			   	  [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])
	N = np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948])
	K = get_num_connections(P, N, N.T)
	W = J*K

	Iext = np.repeat(params['Iext'], num_pops)
	ONEHZ = params['Iext']/8

	stim = np.zeros_like(Iext)
	stim[2] += ONEHZ*20
	stim[3] += ONEHZ*20

	stim = {'onset':  10,	# Stimulus onset (in seconds)
			'offset': 13, 	# Stimulus offset (in seconds)
			'Iext': stim	# Stimulus amplitude
			}

	Kext = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])
	Wext = Kext * J_E

	T = 30
	dt = 1e-3
	I, R, S = DMF(W, Wext, Iext, params=params, stim=stim, t_sim=T, dt=dt)

	# Simulation results
	np.save('DMF/I.npy', I)	# I: Input current
	np.save('DMF/R.npy', R)	# R: Firing rate
	np.save('DMF/S.npy', S)	# S: Synaptic gating

	# Simulation parameters
	P = {'T': 		T,				# Simulation time (in seconds)
		 'dt': 		dt,				# Integration step (in seconds)
		 'K':   	4, 				# Number of layers
		 'onset': 	stim['onset'],	# Stimulus onset (in seconds)
		 'offset': 	stim['offset'],	# Stimulus offset (in seconds)
		 'Iext': 	stim['Iext']	# Stimulus amplitude
		 }	
	np.save('DMF/P.npy', P)