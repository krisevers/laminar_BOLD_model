import numpy as np
import pylab as plt
import matplotlib as mpl

import IPython

from DMF_to_CBF import *

from LBR import *

# Laminar BOLD response
#==========================================================================
# Specify neuronal and NVC model:
#--------------------------------------------------------------------------
A = np.load('DMF/S.npy', allow_pickle=True)			# Neural activity from dynamic mean field model
P = np.load('DMF/P.npy', allow_pickle=True).item()	# Parameters of neural simulation

T  = P['T']						# Total length of the response (in seconds)
dt = P['dt']					# Integration step (in seconds)
K  = P['K']						# Number of depths
onset = int(P['onset']/dt)		# Stimulus onset
offset = int(P['offset']/dt)	# Stimulus offset

P         	= DMF_to_CBF_parameters(P) 	        # Get default parameters (see inside the function)
neuro, cbf  = DMF_to_CBF_model(A, P) 			# Generate the cerebral blood flow response (CBF)


# Specify LBR model:
#--------------------------------------------------------------------------  
P         = LBR_parameters(K,P)   				# get default parameters (see inside the function), 
                               					# NOTE: By default baseline CBV is increasing towards the surface in the ascending vein

P['alpha_v']   = 0.35   	# Choose steady-state CBF-CBV coupling for venules
P['alpha_d']   = 0.2        # Choose steady-state CBF-CBV coupling for ascending vein
P['tau_d_de']  = 30         # Choose dynamic CBF-CBV uncoupling for ascending vein

LBR, LBRpial, Y = LBR_model(P,cbf);  # Generate the laminar bold response


time_axis = np.arange(0, P['T'], P['dt']) - onset*P['dt']	# time axis in seconds

colors = plt.cm.Spectral(np.linspace(0,.3,K))
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

# Display underlying physiological responses
plt.figure(figsize=(10, 5))
plt.subplot(2, 3, 1) 
plt.plot(time_axis,cbf)
plt.xlim(left=time_axis[1], right=time_axis[-1])
#plt.ylim(bottom=0.8, top=1.6)
plt.xlabel('Time (s)')
plt.ylabel('Relative CBF in MV (%)')
plt.subplot(2, 3, 2)
plt.plot(time_axis,Y['mv'])
plt.xlim(left=time_axis[1], right=time_axis[-1])
#plt.ylim(bottom=0.8, top=1.6)
plt.xlabel('Time (s)')
plt.ylabel(r'Relative $CMRO_2$ in MV (%)')
plt.subplot(2, 3, 3)
plt.plot(time_axis,Y['vv'])
plt.xlim(left=time_axis[1], right=time_axis[-1])
#plt.ylim(bottom=0.8, top=1.6)
plt.xlabel('Time (s)')
plt.ylabel('Relative CBV in MV (%)')
plt.subplot(2, 3, 4)
plt.plot(time_axis,Y['qv'])
plt.xlim(left=time_axis[1], right=time_axis[-1])
#plt.ylim(bottom=0.7, top=1.2)
plt.xlabel('Time (s)')
plt.ylabel('Relative dHb in MV (%)')
plt.subplot(2, 3, 5)
plt.plot(time_axis,Y['vd'])
plt.xlim(left=time_axis[1], right=time_axis[-1])
#plt.ylim(bottom=0.8, top=1.6)
plt.xlabel('Time (s)')
plt.ylabel('Relative CBV in AV (%)')
plt.subplot(2, 3, 6)
p = plt.plot(time_axis,Y['qd'])
plt.xlim(left=time_axis[1], right=time_axis[-1])
#plt.ylim(bottom=0.8, top=1.6)
plt.xlabel('Time (s)')
plt.ylabel('Relative dHb in AV (%)')
plt.legend([p[0], p[1], p[2], p[3]],['L23','L4', 'L5', 'L6'])
plt.tight_layout(pad=1)
plt.savefig('svg/physiological_responses.svg')

# Display laminar BOLD response
plt.figure(figsize=(10, 5)),
plt.subplot(1, 3, 1), 
p = plt.plot(time_axis,LBR)
plt.xlim(left=time_axis[1], right=time_axis[-1])
plt.ylim(bottom=-1, top=4)                         
plt.xlabel('Time (s)')
plt.ylabel('LBR (%)')
plt.legend([p[0], p[1], p[2], p[3]],['L23','L4', 'L5', 'L6'])



#calculate time to peak (TTP) and time to undershoot (TTU) with respect to
#the stimulus onset and offset, respectively
TTP   	 = np.zeros(K)
TTU      = np.zeros(K)
for i in range(K):
	Peak_Amp = np.max(LBR[onset:-1,i])
	Peak_Pos = np.where(LBR[onset:-1,i] == np.max(LBR[onset:-1,i]))[0][0]
	PSU_Amp  = np.min(LBR[onset:-1,i])
	PSU_Pos  = np.where(LBR[onset:-1,i] == np.min(LBR[onset:-1,i]))[0][0]

	TTP[i] = time_axis[onset+Peak_Pos].T
	TTU[i] = time_axis[offset+PSU_Pos].T-(offset-onset)*P['dt']

# Display TTP and TTU as function of cortical depth
plt.subplot(1, 3, 2)
plt.plot(P['l'],np.flipud(TTP),'.-', color='black')
plt.xlim(left=0, right=100)
plt.ylim(bottom=0, top=15)                         
plt.xlabel('1 - Cortical depth (%)')
plt.ylabel('TTP (s)')
plt.subplot(1, 3, 3)
plt.plot(P['l'],np.flipud(TTU),'.-', color='black')
plt.xlim(left=0, right=100) 
plt.ylim(bottom=0, top=15)                      
plt.xlabel('1 - Cortical depth (%)')
plt.ylabel('TTU (s)')
plt.tight_layout(pad=1)
plt.savefig('svg/laminar_BOLD_response.svg')

plt.show()