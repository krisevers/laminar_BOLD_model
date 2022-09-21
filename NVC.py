import numpy as np
import copy

import IPython

def NVC_model(neuro, P):
    """
    INPUT:
        K - Number of cortical depths

    OUTPUT:
        Y - structure with all baseline and relative physiological variables

    AUTHOR: Martin Havlicek, 5 August, 2019
    """

    K = P['K']

    # Neuronal parameters:
    # --------------------------------------------------------------------------
    sigma = P['sigma']  # self-inhibitory connection
    mu = P['mu']  # inhibitory-excitatory connection
    lambda_ = P['lambda']  # inhibitory gain
    Bsigma = P['Bsigma']  # modulatory parameter of self-inhibitory connection
    Bmu = P['Bmu']  # modulatory parameter of inhibitory-excitatory connection
    Blambda = P['Blambda']  # modulatory parameter of inhibitory connection
    C = P['C']
    # NVC parameters:
    # --------------------------------------------------------------------------
    c1 = P['c1']
    c2 = P['c2']
    c3 = P['c3']

    # Initial condtions:
    Xn = np.zeros((K, 4))
    yn = np.zeros((K, 4))

    dt = P['dt']
    neuro = np.zeros((int(P['T'] / dt), K))
    cbf = np.zeros((int(P['T'] / dt), K))
    for t in range(int(P['T'] / dt)):
        Xn[:, 3] = np.exp(Xn[:, 3])
        # ----------------------------------------------------------------------
        # Vasoactive signal:
        yn[:, 2] = yn[:, 2] + dt * (Xn[:, 0] - c1 * (Xn[:, 2]))
        # ----------------------------------------------------------------------
        # Inflow:
        df_a = c2 * Xn[:, 2] - c3 * (Xn[:, 3] - 1)
        yn[:, 3] = yn[:, 3] + dt * (df_a / Xn[:, 3])

        Xn = copy.deepcopy(yn)

        cbf[t, :] = np.exp(yn[:, 3]).T
        neuro[t, :] = yn[:, 0].T

    return neuro, cbf