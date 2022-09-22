function cbf = NVC_model(X, dt)
% INPUTS:
%       X  - Neuronal response with dimensions [time, populations]
%       dt - Time step size (in s)
% OUTPUTS:
%       cbf - Cerebral Blood Flow

[T, M] = size(X);   % number of populations

% NVC parameters:
% --------------------------------------------------------------------------
c1      = 0.6;
c2      = 1.5;
c3      = 0.6;

% Initial condtions:
Xvaso   = zeros(M,1);
Yvaso   = zeros(M,1);
Xinflow = zeros(M,1);
Yinflow = zeros(M,1);

cbf   = zeros(T,M);
for t = 1:T
    Xinflow(:) = exp(Xinflow(:));
    %----------------------------------------------------------------------
    % Vasoactive signal:
    Yvaso(:) = Yvaso(:) + dt * (X(t,:)' - c1.*Xvaso(:));
    %----------------------------------------------------------------------
    % Inflow:
    df_a      = c2.*Xvaso(:) - c3.*(Xinflow(:)-1);
    Yinflow(:)   = Yinflow(:) + dt*(df_a./Xinflow(:));
    
    Xvaso(:)         = Yvaso(:);
    Xinflow(:)       = Yinflow(:);
     
    cbf(t,:)   = exp(Yinflow(:))';
end