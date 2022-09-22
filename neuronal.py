import numpy as np
import copy

def neuronal_model(U, P):

    K = P['K']

    # Neuronal parameters:
    # --------------------------------------------------------------------------
    C = P['C']                      # external connection

    # Initial condtions:
    Xn = np.zeros(K*2)
    yn = np.zeros(K*2)

    dt = P['dt']
    neuro = np.zeros((int(P['T'] / dt), K*2))

    for t in range(int(P['T'] / dt)):

        yn = yn + dt * (np.dot(P['W'], Xn) + np.dot(C, U['u'][t, :].T))

        Xn = copy.deepcopy(yn)

        neuro[t, :] = yn.T

    return neuro

def neuronal_parameters(K, P):
    P['K'] = K

    if K < 10:
        P['dt'] = 0.01  # default integration step
    elif K < 20:
        P['dt'] = 0.005  # smaller for higher number of cortical depths
    else:
        P['dt'] = 0.001

    # Neuronal parameter:
    # --------------------------------------------------------------------------
    sigma = -3
    mu = 1.5
    lambda_ = 0.2
    P['C'] = np.eye(K*2)

    W = np.array([[sigma, -mu],
                  [lambda_, -lambda_]])
    Z = np.zeros((2, 2))
    P['W'] = np.asarray(np.bmat([[W, Z, Z, Z],
                                 [Z, W, Z, Z],
                                 [Z, Z, W, Z],
                                 [Z, Z, Z, W]]))

    return P
