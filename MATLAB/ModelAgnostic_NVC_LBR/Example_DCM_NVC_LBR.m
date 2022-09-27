close all; clear all;

%==========================================================================
% Neural parameters and model:
%--------------------------------------------------------------------------
P.K = 4;
P.M = 2*P.K;

P = DMF_parameters(P);

P.W_rec = ...
         [0.51  -1.000 0.454 -0.433 0.037 -0.000 0.025 -0.000;
          0.195 -0.225 0.046 -0.076 0.025 -0.000 0.004 -0.000;
          0.039 -0.034 0.274 -0.780 0.008 -0.000 0.164 -0.000;
          0.091 -0.004 0.111 -0.234 0.001 -0.000 0.099 -0.000;
          0.119 -0.081 0.062 -0.007 0.023 -0.108 0.016 -0.000;
          0.014 -0.008 0.007 -0.001 0.004 -0.019 0.001 -0.000;
          0.052 -0.025 0.075 -0.059 0.046 -0.014 0.094 -0.485;
          0.025 -0.001 0.002 -0.000 0.004 -0.001 0.032 -0.061];

P.C = 1.0;

P.T  = 5;
P.dt = 1e-4;

nu_ext = 8;
K_ext = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948];
W_ext = K_ext' * P.J_E;

U = ones(P.M, P.T/P.dt) .* W_ext * nu_ext;
dur    = 3/P.dt;
onset  = 1/P.dt;
offset = onset + dur;

K_inp = 105.3605 * 100;
W_inp = K_inp .* P.J_E;

U(1,onset:offset) = nu_ext * W_ext(1) + 0.00  * W_inp;
U(2,onset:offset) = nu_ext * W_ext(2) + 0.00  * W_inp;
U(3,onset:offset) = nu_ext * W_ext(3) + 20.00 * W_inp;
U(4,onset:offset) = nu_ext * W_ext(4) + 20.00 * W_inp;
U(5,onset:offset) = nu_ext * W_ext(5) + 0.00  * W_inp;
U(6,onset:offset) = nu_ext * W_ext(6) + 0.00  * W_inp;
U(7,onset:offset) = nu_ext * W_ext(7) + 20.00 * W_inp;
U(8,onset:offset) = nu_ext * W_ext(8) + 20.00 * W_inp;

X = DMF_model(P, U);

%==========================================================================
% NVC parameters and model:
%--------------------------------------------------------------------------
E = X(:,1:2:end);   % select excitatory populations

cbf = NVC_model(E, P.dt);

time_axis = [0:P.dt:P.T-P.dt]; %- onset*P.dt; % time axis in seconds

%%% plotting
figure(1)
hold on
subplot(211), plot(time_axis, X),   ylabel('Neuronal Response'),   xlabel('Time (s)'), xlim([time_axis(1), time_axis(end)]);
subplot(212), plot(time_axis, cbf), ylabel('Cerebral Blood Flow'), xlabel('Time (s)'), xlim([time_axis(1), time_axis(end)]);
hold off
  