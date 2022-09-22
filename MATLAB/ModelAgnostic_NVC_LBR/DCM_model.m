function X = DCM_model(P,U)

M = P.M;    % Number of populations

% Neuronal parameter:
%--------------------------------------------------------------------------
C = P.C;    % External weight

A = P.A;    % Connection weights 

% Initial condtions:
Xn  = zeros(M,1);
yn  = Xn;

T   = P.T;
dt  = P.dt;
X = zeros(T/dt,M);
for t = 1:T/dt
    % Neuronal (excitatory & inhibitory)
    yn = yn + dt * (A * Xn + C * U(:,t));
    
    Xn = yn;
     
    X(t,:) = yn';
end