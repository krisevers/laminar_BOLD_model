import numpy as np
import pylab as plt

import neuronal_NVC_model
import neuronal_NVC_parameters

import LBR_model
import LBR_parameters

K = 6 	# number of depths

# Laminar BOLD response to short 2 sec stimulus
#==========================================================================
# Specify neuronal and NVC model:
#--------------------------------------------------------------------------
P         = neuronal_NVC_parameters(K) 			# get default parameters (see inside the function)
P['T']    = 30                					# Total lenght of the response (in seconds)
dur       = 2/P['dt']        					# Stimulus duration (in second, e.g. 2 sec) ... dt - refers to integration step
onset     = 3/P['dt']         					# Stimulus onset time (in seconds) 
offset    = onset + dur     					# Stimulus offset time (in seconds) 
U['u']       = np.zeros(int(P['T']/P['dt']),K)  # Matrix with input vectors to the neuronal model (one column per depth)
U['u'][onset:offset,:] = 1             			# Set one during stimulus window
neuro, cbf  = neuronal_NVC_model(P,U) 			# Generate the neuronal and cerebral blood flow response (CBF)
  
# Specify LBR model:
#--------------------------------------------------------------------------  
P         = LBR_parameters(K,P)   				# get default parameters (see inside the function), 
                               					# NOTE: By default baseline CBV is increasing towards the surface in the ascending vein
  
P['alpha_v']   = 0.35   	# Choose steady-state CBF-CBV coupling for venules
P['alpha_d']   = 0.2        # Choose steady-state CBF-CBV coupling for ascending vein
P['tau_d_de']  = 30         # Choose dynamic CBF-CBV uncoupling for ascending vein

LBR, Y = LBR_model(P,cbf);  # Generate the laminar bold response


# time_axis = [0:P['dt']:P['T']-P['dt']] - onset*P['dt'] 	# time axis in seconds



# # Display underlying physiological responses
# plt.figure()
# plt.subplot(2, 3, 1) 
# plt.plot(time_axis,cbf)
# plt.xlim(left=time_axis[0], right=time_axis[-1])
# plt.ylim(bottom=0.8, top=1.6)
# plt.xlabel('Time (s)')
# plt.ylabel('Relative CBF in MV (%)')
# plt.subplot(2, 3, 2)
# plt.plot(time_axis,Y['mv'])
# plt.xlim(left=time_axis[0], right=time_axis[-1])
# plt.ylim(bottom=0.8, top=1.6)
# plt.xlabel('Time (s)')
# plt.ylabel('Relative CMRO_2 in MV (%)')
# plt.subplot(2, 3, 3)
# plt.plot(time_axis,Y['vv'])
# plt.xlim(left=time_axis[0], right=time_axis[-1])
# plt.ylim(bottom=0.8, top=1.6)
# plt.xlabel('Time (s)')
# plt.ylabel('Relative CBV in MV (%)')
# plt.subplot(2, 3, 4)
# plt.plot(time_axis,Y['qv'])
# plt.xlim(left=time_axis[0], right=time_axis[-1])
# plt.ylim(bottom=1.2, top=0.7)
# plt.xlabel('Time (s)')
# plt.ylabel('Relative dHb in MV (%)')
# plt.subplot(2, 3, 5)
# plt.plot(time_axis,Y['vd'])
# plt.xlim(left=time_axis[0], right=time_axis[-1])
# plt.ylim(bottom=0.8, top=1.6)
# plt.xlabel('Time (s)')
# plt.ylabel('Relative CBV in AV (%)')
# plt.subplot(2, 3, 6)
# p = plt.plot(time_axis,Y['qd'])
# plt.xlim(left=time_axis[0], right=time_axis[-1])
# plt.ylim(bottom=0.8, top=1.6)
# plt.xlabel('Time (s)')
# plt.ylabel('Relative dHb in AV (%)')
# # plt.legend([p[0] p[-1]],{'Upper','Lower'})
 

# # Display laminar BOLD response
# plt.figure(),
# plt.subplot(1, 3, 1), 
# p = plt.plot(time_axis,LBR)
# plt.xlim(left=time_axis[0], right=time_axis[-1])
# plt.ylim(bottom=-1, top=4)                         
# plt.xlabel('Time (s)')
# plt.ylabel('LBR (%)')
# # plt.legend([p(1) p(end)],{'Upper','Lower'});

# # calculate time to peak (TTP) and time to undershoot (TTU) with respect to
# # the stimulus onset and offset, respectively
# [Peak_Amp,Peak_Pos] = np.max(LBR[onset:-1,:])
# [PSU_Amp,PSU_Pos]   = np.min(LBR[offset:-1,:])
# TTP = time_axis[onset+Peak_Pos].T
# TTU = time_axis[offset+PSU_Pos].T-(offset-onset)*P['dt']
# # Display TTP and TTU as function of cortical depth
# plt.subplot(1, 3, 2)
# plt.plot(P['l'],np.flipud(TTP),'.-')
# plt.xlim(left=0, right=100)
# plt.ylim(bottom=0, top=12)                         
# plt.xlabel('1 - Cortical depth (%)')
# plt.ylabel('TTP (s)')
# plt.subplot(1, 3, 3), plot(P['l'],flipud(TTU),'.-')
# plt.xlim(left=0, right=100) 
# plt.ylim(bottom=0, top=12)                      
# plt.xlabel('1 - Cortical depth (%)')
# plt.ylabel('TTU (%)')