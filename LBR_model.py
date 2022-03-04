# LBR model
import numpy as np

def LBR_model(P, cbf, *args):
	'''
	INPUTS:
		P - structure of model parameters
		cbf - matrix defining laminar cerebral blood flow (CBF) response, [time, depht]
		cmro2 - matrix defining the laminar changes in oxygen metabolism (CMRO2), [time, depth]

	OUTPUTS:
		LBR - matrix containing laminar BOLD responses in percent signal change [time, depth]
		Y - structure with all baseline and relative physiological variables underlying BOLD response
		LBRpial - BOLD response of the pial vein in percent signal change (0th depth) [time, 1]

	AUTHOR: Martin Havlicek, 5 August, 2019
	'''

	if len(args) < 1:
		cmro2 = []

	##
	# Hemodynamic model parameters
	#------------------------------------------------------
	K = P['K']		# Number of depths

	# BASELINE PARAMETERS
	V0t 	= P['V0t']		# Total amount of CBV0 within GM tissue (in mL)
	V0t_p	= P['V0t_p']	# Total amount of CBV0 in pial vein (in mL)

	w_v		= P['w_v']		# Fraction of CBV0 in venules with respect to the total
	w_d  	= 1-w_v			# Fraction of CBV0 in ascending vein with respect to the total

	s_v 	= P['s_v']		# Slope of CBV0 increase towards the surface in venules
	s_d  	= P['s_d']		# Slope of CBV0 increase towards the surface in ascending veins

	# Depth-specific CBV0
	if len(P['x_v']) == K:              # For venules
		x_v  = P['x_v']                	# Depth-specific fractions defined by user
	else:
	    x_v  = 10+s_v*np.flipud(P['l']) # Possibility to define linear increase (default s_v = 0)

	x_v      = x_v/np.sum(x_v)          # Fraction of CBV0 across depths in venules 

	if len(P['x_v']) == K:              # For ascending vein
	    x_d  = P['x_d']                 # Depth-specific fractions defined by user
	else:
	    x_d  = 10+s_d*np.flipud(P['l']) # Possibility to define linear increase 
	x_d      = x_d/np.sum(x_d)          # Fraction of CBV0 across depths in venules 

	V0v      = V0t*w_v*x_v              # CBV0 in venules
	V0d      = V0t*w_d*x_d              # CBV0 in ascending vein
	V0p      = V0t_p                    # CBV0 in pial vein

	# Transit time through venules (or microvasculature in general)
	if len(P['t0v']) == K:
		t0v = P['t0v']
	else:
		t0v = np.ones(K)*P['t0v']

	# Depth-specific baseline CBF
	F0v = V0v/t0v
	F0d = np.flipud(np.cumsum(np.flipud(F0v)))
	F0p = F0d[1]

	# Depth-specific transit time
	t0v = V0v/F0v
	t0d = V0d/F0d
	t0p = V0p/F0p

	# Total mean transit time
	tt0v = np.mean(t0v)
	tt0d = np.mean(np.cumsum(t0d))
	tt0  = tt0v + tt0d

	# Baseline oxygen extraction fraction
	if len(P['E0v']) == K:
		E0v        = P['E0v']     # depth-specific defined by user
	else:
		E0v        = np.ones(K)*P['E0v']
	if lenn(P['E0d']) == K:
		E0d        = P['E0d']      # depth-specific defined by user
	else:
		E0d        = np.ones(K)*P['E0d']
	E0p        = P['E0p']


	# PARAMETERS DESCRIBING RELATIVE RELATIONSHIPS BETWEEN PHYSIOLOGICAL VARIABLES:
	# n-ratio (= (cbf-1)./(cmro2-1)). Not used if cmro2 response is directly specified as an input
	if len(P['n']) == K:             	# For venules (microvasculature)
	    n      = P['n']                	# Depth-specific defined by user
	else:
	    n      = np.ones(K)*P['n']      # Default

	# Grubb's exponent alpha (i.e CBF-CBV steady-state relationship)
	if len(P['alpha_v']) == K:       			# For venules
	    alpha_v    = P['alpha_v']       		# Depth-specific defined by user 
	else:
	    alpha_v    = np.ones(K)*P['alpha_v']  	# Default
	if len(P['alpha_d']) == K:       			# For ascending vein
	    alpha_d    = P['alpha_d']             	# Depth-specific defined by user  
	else:
	    alpha_d    = np.ones(K)*P['alpha_d']  	# Default
	alpha_p        = P['alpha_p']      			# For pial vein

	# CBF-CBV uncoupling (tau) during inflation and deflation:
	if len(P['tau_v_in']) == K:      			# For venules (inflation)
	    tau_v_in  = P['tau_v_in']             	# Depth-specific defined by user
	else:
	    tau_v_in  = np.ones(K)*P['tau_v_in']  	# Default  
	if len(P['tau_v_de']) == K:      			# For venules (deflation)
	    tau_v_de  = P['tau_v_de']             	# Depth-specific defined by user  
	else:
	    tau_v_de  = np.ones(K)*P['tau_v_de']  	# Default  
	if len(P['tau_d_in']) == K:      			# For ascending vein (inflation)
	    tau_d_in  = P['tau_d_in']	            # Depth-specific defined by user 
	else:
	    tau_d_in  = np.ones(K)*P['tau_d_in']   	# Default  
	if len(P['tau_d_de']) == K:       			# For ascending vein (deflation)
	    tau_d_de  = P['tau_d_de']             	# Depth-specific defined by user 
	else:
	    tau_d_de  = np.ones(K)*P['tau_d_de']  	# Default
	tau_p_in      = P['tau_p_in']       		# For pial vein (inflation)
	tau_p_de      = P['tau_p_de']       		# For pial vein (deflation)



	##
	# Parameters for laminar BOLD signal equation (for 7 T field strenght):
	#------------------------------------------------------
	# Baseline CBV in fraction with respect to GM tissue
	V0vq = V0v/100*K
	V0dq = V0d/100*K
	V0pq = V0p/100*K

	TE     = P['TE']	 	# echo-time (sec) 

	Hct_v  = P['Hct_v']		# Hematocrit fraction
	Hct_d  = P['Hct_d']
	Hct_p  = P['Hct_p']
	B0     = P['B0']   		# Field strenght        
	gyro   = P['gyro']      # Gyromagnetic constant 
	suscep = P['suscep']    # Susceptibility difference

	nu0v   = suscep*gyro*Hct_v*B0
	nu0d   = suscep*gyro*Hct_d*B0
	nu0p   = suscep*gyro*Hct_p*B0 

	# Water proton density 
	rho_t  = P['rho_t']  # In GM tissue
	rho_v  = P['rho_v']  # In blood (venules) Ref. Lu et al. (2002) NeuroImage
	rho_d  = P['rho_d']  # In blood (ascening vein) 
	rho_p  = P['rho_p']  # In blood (pial vein) 
	rho_tp = P['rho_tp'] # In in tissue and CSF 

	# Relaxation rates (in sec-1):
	if len(P['R2s_t']) == K:  	# For tissue
	    R2s_t  = P['R2s_t']
	else:
	    R2s_t  = np.ones(K)*P['R2s_t']   	# (sec-1)
	if len(P.R2s_v) == K:  					# For venules
	    R2s_v  = P['R2s_v']               	# (sec-1)
	else:
	    R2s_v  = np.ones(K)*P['R2s_v']  	# (sec-1) 
	if len(P['R2s_d']) == K:  				# For ascening vein
	    R2s_d  = P['R2s_d']           		# (sec-1)
	else:
	    R2s_d  = np.ones(K)*P['R2s_d'] 		# (sec-1)  
	R2s_p  = P['R2s_p']         			# For pial vein 

	# (Baseline) Intra-to-extra-vascular signal ratio
	ep_v   = rho_v/rho_t*np.exp(-TE*R2s_v)/np.exp(-TE*R2s_t) 	# For venules
	ep_d   = rho_d/rho_t*np.exp(-TE*R2s_d)/np.exp(-TE*R2s_t)	# For ascending vein
	ep_p   = rho_p/rho_tp*np.exp(-TE*R2s_p)/np.exp(-TE*R2s_t)	# For pial vein 

	# Slope of change in R2* of blood with change in extraction fration during activation 
	r0v    = 228	# For venules   
	r0d    = 232    # For ascending vein
	r0p    = 236    # For pial vein

	H0     = 1/(1 - V0vq - V0dq + ep_v*V0vq + ep_d*V0dq)	# constant in front
	H0p    = 1/(1 - V0pq + ep_p*V0pq)

	k1v     = 4.3*nu0v*E0v*TE
	k2v     = ep_v*r0v*E0v*TE
	k3v     = 1 - ep_v

	k1d     = 4.3*nu0d*E0d*TE
	k2d     = ep_v*r0d*E0d*TE
	k3d     = 1 - ep_d

	k1p     = 4.3*nu0p*E0p*TE
	k2p     = ep_p*r0p*E0p*TE
	k3p     = 1 - ep_p



	##
	# Initial conditions
	#------------------------------------------------------
	Xk       = np.zeros((K,4))
	Xp       = np.zeros((1,2))

	yk       = Xk
	yp       = Xp

	f_d      = np.ones(K)
	dv_d     = np.ones(K)
	dHb_d    = np.ones(K)

	tau_v    = tau_v_in
	tau_d    = tau_d_in
	tau_p    = tau_p_in

	# integration step
	dt = P['dt']

	LBR       = np.zeros((int(P['T']/dt),K))
	LBRpial   = np.zeros((int(P['T']/dt),K))


	##
	# Simulation
	#------------------------------------------------------
	for t in range(1, int(P['T']/dt)):

	    Xk      = np.exp(Xk)    # log-normal transformation (Stephan et al.(2008), NeuroImage)
	    Xp      = np.exp(Xp)
	    
	    # model input (laminar CBF response):
	    f_a = cbf[t,:].T
	    
	    # VENULES COMPARTMENTS:
	    #--------------------------------------------------------------------------
	    # blood outflow from venules compartment
	    if np.sum(alpha_v)>0:
	    	f_v     = (V0v*Xk[:,0]**(1/alpha_v) + F0v*tau_v*f_a)/(V0v+F0v*tau_v)
	    else:
	        f_v     = f_a
	    
	    # change in blood volume in venules:
	    dv_v        = (f_a - f_v)/t0v
	    # change in oxygen matabolims (CMRO2)
	    if isempty(cmro2):
	        m        = (f_a + n-1)/n  # (if not specified directly)
	    else:
	        m        = cmro2[t,:].T

	    # change in deoxyhemoglobin content venules:
	    dHb_v        = (m - f_v*Xk[:,1]./Xk[:,0])/t0v


	    # ASCENDING VEIN COMPARTMENTS:
	    #--------------------------------------------------------------------------    
	    # blood outflow from Kth depth of ascending vein compartment (deepest depth):
	    if alpha_d[-1]>0:
	        f_d[-1]  = (V0d[-1]*Xk[end,2]**(1/alpha_d[-1]) + tau_d[-1]*f_v[-1]*F0v[-1])/(V0d[-1]+F0d[-1]*tau_d[-1])
	    else:
	        f_d[-1]  = f_v[-1]*F0v[-1]/F0d[-1]

	    # changes in blood volume and deoxyhemoglobin in ascending vein (deepest depth):
	    dv_d[-1]     = (f_v[-1] - f_d[-1])./t0d[-1]
	    dHb_d[-1]    = (f_v[-1]*Xk[-1,1]/Xk[-1,0] - f_d[-1]*Xk[-1,3]/Xk[-1,2])/t0d[-1]
	    
	    # blood outflow from other comparments of ascending vein:
	    for i in range(K-1, 1, -1):
	        if alpha_d[i]>0:
	            f_d[i]     = (V0d[i]*Xk[i,2]**(1/alpha_d[i]) + tau_d[i]*(f_v[i]*F0v[i]+f_d[i+1]*F0d[i+1]))/(V0d[i]+F0d[i]*tau_d[i])
	        else:
	            f_d[i]     = f_v[i]*F0v[i]/F0d[i]+f_d[i+1]*F0d[i+1]/F0d[i]
	        
	        # changes in blood volume and deoxyhemoglobin in ascending vein:
	        dv_d[i]    = (f_v[i]*F0v[i]/F0d[i] + f_d[i+1]*F0d[i+1]/F0d[i] - f_d[i])/t0d[i]
	        dHb_d[i]   = (f_v[i]*F0v[i]/F0d[i]*Xk[i,1]/Xkp[i,0] + f_d[i+1]*F0d[i+1]/F0d[i]*Xk[i+1,3]/Xk[i+1,2] - f_d[i]*Xk[i,3]/Xk[i,2])/t0d[i]

	    
	    # PIAL VEIN COMPARTMENT:
	    #--------------------------------------------------------------------------    

	    # blood outflow from pial vein:
	    if alpha_p>0:
	        f_p     = (V0p*Xp[0]^(1/alpha_p) + F0p*tau_p*f_d[0])/(V0p+F0p*tau_p)
	    else:
	        f_p     = f_d[0]
	    
	    # changes in blood volume and deoxyhemoglobin in pial vein:
	    dv_p  = (f_d[0] - f_p)/t0p
	    dHb_p = (f_d[0]*Xk[0,3]/Xk[0,2] - f_p*Xp[1]/Xp[0])/t0p
	    
	    
	    # Intergrated changes to previous time point
	    yk[:,0]  = yk[:,0] + dt*(dv_v/Xk[:,0])
	    yk[:,1]  = yk[:,1] + dt*(dHb_v/Xk[:,1])
	    yk[:,2]  = yk[:,2] + dt*(dv_d/Xk[:,2])
	    yk[:,3]  = yk[:,3] + dt*(dHb_d/Xk[:,3])
	    
	    yp[:,0]  = yp[:,0] + dt*(dv_p/Xp[0])
	    yp[:,1]  = yp[:,1] + dt*(dHb_p/Xp[1])

	    Xk        = yk
	    Xp        = yp
	    
	    tau_v     = tau_v_in
	    tau_d     = tau_d_in
	    tau_p     = tau_p_in
	 
	    # check for deflation (negative derivative)
	    tau_v[dv_v<0]  = tau_v_de[dv_v<0]
	    tau_d[dv_d<0]  = tau_d_de[dv_d<0]
	    tau_p[dv_p<0]  = tau_p_de[dv_p<0]
	    
	    # venules:
	    m_v  = m;
	    v_v  = np.exp(yk[:,0])		# log-normal transformation
	    q_v  = np.exp(yk[:,1])
	    # draining vein:
	    v_d  = np.exp(yk[:,2])
	    q_d  = np.exp(yk[:,3])
	    # pail vein:
	    v_p  = np.exp(yp[:,0])
	    q_p  = np.exp(yp[:,1])
	    
	    
	    # save physiological variable:
	    Y['fa'][t,:] = f_a
	    Y['mv'][t,:] = m_v
	    Y['qv'][t,:] = q_v
	    Y['qd'][t,:] = q_d
	    Y['qp'][t,:] = q_p

	    Y['vv'][t,:] = v_v
	    Y['vd'][t,:] = v_d
	    Y['vp'][t,:] = v_p

	    
	    
	    LBR[t,:] = H0*((1-V0vq-V0dq)*(k1v*V0vq*(1-q_v) +k1d*V0dq*(1-q_d)) + 
	                                    + k2v*V0vq*(1-q_v/v_v) + k2d*V0dq*(1-q_d/v_d) +
	                                    + k3v*V0vq*(1-v_v)     + k3d*V0dq*(1-v_d))*100
	    
	    
	    LBRpial[t,:] = H0p*((1-V0pq)*(k1p*V0pq*(1-q_p)) + k2p*V0pq*(1-q_p/v_p) +
	                                    				  k3p*V0pq*(1-v_p))*100


	# save baseline physiological parameters
	Y['F0v']  = F0v
	Y['F0d']  = F0d
	Y['F0p']  = F0p

	Y['V0v']  = V0v
	Y['V0d']  = V0d
	Y['V0p']  = V0p

	Y['V0vq'] = V0vq
	Y['V0dq'] = V0dq
	Y['V0pq'] = V0pq

	Y['t0v']  = t0v
	Y['t0d']  = t0d
	Y['t0p']  = t0p
	Y['tt0v'] = tt0v
	Y['tt0d'] = tt0d
	Y['tt0']  = tt0

	return LBR, LBRpial, Y