J_E = 87.8e-3;    % synaptic strength (nA)
sigma = 0.02;     % std input noise (nA)
g = -4;           % relative inhibitory synaptic strength
tau_r = 2e-3;     % refractory period
tau_s = 0.5e-3;   % postsynaptic current time constant (s)
tau_m = 10e-3;    % membrane time constant (s)
C_m = 250e-6;     % membrane capacitance (mF)
R = tau_m / C_m;  % membrane resistance


%%% gain function and parameters
f = @(h) (a * h - b) ./ (1 - exp(-d * (a * h - b)));
a = 48;
b = 981;
d = 8.9e-3;

%%% external input
K_ext = ...   % number of external connections
    [1600;...
    1500;...
    2100;...
    1900;...
    2000;...
    1900;...
    2900;...
    2100];

nu_ext = ones(8,1) * 8; % firing rate per connection (Hz)

%%% recurrent connectivity
N = ...                 % number of neurons
    [20683;...
    5834;...
    21915;...
    5479;...
    4850;...
    1065;...
    14395;...
    2948];

P = ...                 % connection probability
    [0.101,0.169,0.084,0.082,0.032,0.0,0.008,0.0;...
    0.135,0.137,0.032,0.052,0.075,0.0,0.004,0.0;...
    0.008,0.006,0.050,0.135,0.007,0.0003,0.045,0.0;...
    0.069,0.003,0.079,0.160,0.003,0.0,0.106,0.0;...
    0.100,0.062,0.051,0.006,0.083,0.373,0.020,0.0;...
    0.055,0.027,0.026,0.002,0.060,0.316,0.009,0.0;...
    0.016,0.007,0.021,0.017,0.057,0.020,0.040,0.225;...
    0.036,0.001,0.003,0.001,0.028,0.008,0.066,0.144];

K_hat = ...              % number of connections
    log(1-P) ./ log(1 - 1./(N * N')) ./ N;

J_I = g * J_E;

W_rec = repmat([J_E, J_I],[8,4]) .* K_hat;
W_ext = K_ext * J_E;

%%% simulation (with noise)
t_sim = 1;
dt = 1e-4;
t_steps = floor(t_sim / dt) + 1;

I = zeros(8,t_steps);  % store current for eventually computing BOLD signal
H = zeros(8,t_steps);
F = zeros(8,t_steps);
dsig = sqrt(dt/tau_s) * sigma;
for t=2:t_steps
    I(:,t) = I(:,t-1) + dt * (-I(:,t-1) / tau_s + ...
        W_rec * F(:,t-1) + W_ext .* nu_ext) + dsig * randn(8,1);
    H(:,t) = H(:,t-1) + dt * (-H(:,t-1) + R * I(:,t)) / tau_m;
    F(:,t) = f(H(:,t));
end

%%% plotting
figure()
hold on
for n=1:8
    plot(F(n,:))
end
legend('L23e','L23i','L4e','L4i','L5e','L5i','L6e','L6i')
hold off

figure()
nu_exc = mean(F(end:-2:1,:),2);
nu_inh = mean(F(end-1:-2:1,:),2);
nu_final = [nu_exc,nu_inh];
barh(nu_final)
yticklabels({'L6';'L5';'L4';'L2/3'})
