function [I, H, F] = DMF_model(P, U)
% INPUTS:
%       
% OUTPUTS:
%       cbf - Cerebral Blood Flow
f = @(h) (P.a * h - P.b) ./ (1 - exp(-P.d * (P.a * h - P.b)));

t_steps = floor(P.T / P.dt);

I = zeros(8,t_steps);  % store current for eventually computing BOLD signal
H = zeros(8,t_steps);
F = zeros(8,t_steps);
dsig = sqrt(P.dt/P.tau_s) * P.sigma;
for t=2:t_steps
    I(:,t) = I(:,t-1) + P.dt * (-I(:,t-1) / P.tau_s + ...
        P.W_rec * F(:,t-1) + U(:, t)) + dsig * randn(8,1);
    H(:,t) = H(:,t-1) + P.dt * (-H(:,t-1) + P.R * I(:,t)) / P.tau_m;
    F(:,t) = f(H(:,t));
end