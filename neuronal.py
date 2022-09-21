import numpy as np
import copy

def neuronal_model(U, P):

    K = P['K']

    # Neuronal parameters:
    # --------------------------------------------------------------------------
    sigma = P['sigma']              # self-inhibitory connection
    mu = P['mu']                    # inhibitory-excitatory connection
    lambda_ = P['lambda']           # inhibitory gain
    Bsigma = P['Bsigma']            # modulatory parameter of self-inhibitory connection
    Bmu = P['Bmu']                  # modulatory parameter of inhibitory-excitatory connection
    Blambda = P['Blambda']          # modulatory parameter of inhibitory connection
    C = P['C']                      # external connection

    # weights:     E   I
    W = np.array([[0., 0.],
                  [0., 0.]])

    # Initial condtions:
    Xn = np.zeros((K, 4))
    yn = np.zeros((K, 4))

    dt = P['dt']
    neuro = np.zeros((int(P['T'] / dt), K))

    for t in range(int(P['T'] / dt)):
        Xn[:, 3] = np.exp(Xn[:, 3])

        A   = np.eye(K) * sigma
        MU  = np.ones(K) * mu
        LAM = np.ones(K) * lambda_
        for i in range(len(Bsigma)):
            A = A + diag(Bsigma[:, i]) * U['m'][t, i]
        for i in range(len(Bmu)):
            MU = MU + Bmu[:, i] * U['m'][t, i]
        for i in range(len(Blambda)):
            LAM = LAM + Blambda[:, i] * U['m'][t, i]

        # ----------------------------------------------------------------------
        # Neuronal (excitatory & inhibitory)
        yn[:, 0] = yn[:, 0] + dt * (np.dot(A, Xn[:, 0]) - MU * Xn[:, 1] + np.dot(C, U['u'][t, :].T))

        yn[:, 1] = yn[:, 1] + dt * (LAM * (-Xn[:, 1] + Xn[:, 0]))

        Xn = copy.deepcopy(yn)

        neuro[t, :] = yn[:, 0].T

    return neuro