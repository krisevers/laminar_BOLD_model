function X = neuronal_model(P,U)
M      = P.M;

% Neuronal parameter:
%--------------------------------------------------------------------------
C       = P.C;

A       = P.A;

% Initial condtions:
Xn  = zeros(K*2);
yn  = Xn;

dt  = P.dt;
X = zeros(P.T/dt,M);
for t = 1:P.T/dt
    % Neuronal (excitatiry & inhibitory)
    yn = yn + dt * (W * Xn + C * U(:,t));
    
    Xn         = yn;
     
    X(t,:) = yn';
end