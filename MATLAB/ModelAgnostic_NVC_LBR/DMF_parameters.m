function P = DMF_parameters()

g = -4;                 % relative inhibitory synaptic strength
P.J_E = 87.8e-3;        % excitatory synaptic strength (nA)
P.J_I = P.J_E * g;      % inhibitory synaptic strength (nA)
P.sigma = 0.0; %0.02;   % std input noise (nA)
P.tau_r = 2e-3;         % refractory period
P.tau_s = 0.5e-3;       % postsynaptic current time constant (s)
P.tau_m = 10e-3;        % membrane time constant (s)
P.C_m = 250e-6;         % membrane capacitance (mF)
P.R = P.tau_m / P.C_m;  % membrane resistance

%%% gain function and parameters
P.a = 48;
P.b = 981;
P.d = 8.9e-3;