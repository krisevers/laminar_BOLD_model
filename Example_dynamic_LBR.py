import numpy as np
import pylab as plt
import matplotlib as mpl

from neuronal_NVC import *

from LBR import *

K = 13  # number of depths

# Laminar BOLD response to short 2 sec stimulus
# ==========================================================================
# Specify neuronal and NVC model:
# --------------------------------------------------------------------------
P = {}
U = {}
P = neuronal_NVC_parameters(K, P)  				# Get default parameters (see inside the function)
P['T'] = 30  									# Total length of the response (in seconds)
dur = 2 / P['dt']  								# Stimulus duration (in second, e.g. 2 sec) ... dt - refers to integration step
onset = int(3 / P['dt'])  						# Stimulus onset time (in seconds)
offset = int(onset + dur) 			 			# Stimulus offset time (in seconds)
U['u'] = np.zeros((int(P['T'] / P['dt']), K))  	# Matrix with input vectors to the neuronal model (one column per depth)
U['u'][onset:offset, :] = 1  					# Set one during stimulus window
neuro, cbf = neuronal_NVC_model(U, P)  			# Generate the neuronal and cerebral blood flow response (CBF)

plt.figure()
plt.subplot(121)
plt.plot(neuro)
plt.subplot(122)
plt.plot(cbf)
plt.show()

# Specify LBR model:
# --------------------------------------------------------------------------
P = LBR_parameters(K, P)  # get default parameters (see inside the function),
# NOTE: By default baseline CBV is increasing towards the surface in the ascending vein

P['alpha_v'] = 0.35  # Choose steady-state CBF-CBV coupling for venules
P['alpha_d'] = 0.2  # Choose steady-state CBF-CBV coupling for ascending vein
P['tau_d_de'] = 30  # Choose dynamic CBF-CBV uncoupling for ascending vein

LBR, LBRpial, Y = LBR_model(P, cbf)  # Generate the laminar bold response

import IPython; IPython.embed()

time_axis = np.arange(0, P['T'], P['dt']) - onset * P['dt']  # time axis in seconds

colors = plt.cm.Spectral(np.linspace(0, 1, K))
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

# Display underlying physiological responses
plt.figure(figsize=(10, 5))
plt.subplot(2, 3, 1)
plt.plot(time_axis, cbf)
plt.xlim(left=time_axis[1], right=time_axis[-1])
plt.ylim(bottom=0.8, top=1.6)
plt.xlabel('Time (s)')
plt.ylabel('Relative CBF in MV (%)')
plt.subplot(2, 3, 2)
plt.plot(time_axis, Y['mv'])
plt.xlim(left=time_axis[1], right=time_axis[-1])
plt.ylim(bottom=0.8, top=1.6)
plt.xlabel('Time (s)')
plt.ylabel(r'Relative $CMRO_2$ in MV (%)')
plt.subplot(2, 3, 3)
plt.plot(time_axis, Y['vv'])
plt.xlim(left=time_axis[1], right=time_axis[-1])
plt.ylim(bottom=0.8, top=1.6)
plt.xlabel('Time (s)')
plt.ylabel('Relative CBV in MV (%)')
plt.subplot(2, 3, 4)
plt.plot(time_axis, Y['qv'])
plt.xlim(left=time_axis[1], right=time_axis[-1])
plt.ylim(bottom=0.7, top=1.2)
plt.xlabel('Time (s)')
plt.ylabel('Relative dHb in MV (%)')
plt.subplot(2, 3, 5)
plt.plot(time_axis, Y['vd'])
plt.xlim(left=time_axis[1], right=time_axis[-1])
plt.ylim(bottom=0.8, top=1.6)
plt.xlabel('Time (s)')
plt.ylabel('Relative CBV in AV (%)')
plt.subplot(2, 3, 6)
p = plt.plot(time_axis, Y['qd'])
plt.xlim(left=time_axis[1], right=time_axis[-1])
plt.ylim(bottom=0.8, top=1.6)
plt.xlabel('Time (s)')
plt.ylabel('Relative dHb in AV (%)')
plt.legend([p[0], p[-1]], ['Upper', 'Lower'])
plt.tight_layout(pad=1)
plt.savefig('svg/physiological_responses.svg')

# Display laminar BOLD response
plt.figure(figsize=(10, 5)),
plt.subplot(1, 3, 1),
p = plt.plot(time_axis, LBR)
plt.xlim(left=time_axis[1], right=time_axis[-1])
plt.ylim(bottom=-1, top=4)
plt.xlabel('Time (s)')
plt.ylabel('LBR (%)')
plt.legend([p[0], p[-1]], ['Upper', 'Lower']);

# calculate time to peak (TTP) and time to undershoot (TTU) with respect to
# the stimulus onset and offset, respectively
TTP = np.zeros(K)
TTU = np.zeros(K)
for i in range(K):
    Peak_Amp = np.max(LBR[onset:-1, i])
    Peak_Pos = np.where(LBR[onset:-1, i] == np.max(LBR[onset:-1, i]))[0][0]
    PSU_Amp = np.min(LBR[onset:-1, i])
    PSU_Pos = np.where(LBR[onset:-1, i] == np.min(LBR[onset:-1, i]))[0][0]

    TTP[i] = time_axis[onset + Peak_Pos].T
    TTU[i] = time_axis[offset + PSU_Pos].T - (offset - onset) * P['dt']

# Display TTP and TTU as function of cortical depth
plt.subplot(1, 3, 2)
plt.plot(P['l'], np.flipud(TTP), '.-', color='black')
plt.xlim(left=0, right=100)
plt.ylim(bottom=0, top=15)
plt.xlabel('1 - Cortical depth (%)')
plt.ylabel('TTP (s)')
plt.subplot(1, 3, 3)
plt.plot(P['l'], np.flipud(TTU), '.-', color='black')
plt.xlim(left=0, right=100)
plt.ylim(bottom=0, top=15)
plt.xlabel('1 - Cortical depth (%)')
plt.ylabel('TTU (s)')
plt.tight_layout(pad=1)
plt.savefig('svg/laminar_BOLD_response.svg')

plt.show()
