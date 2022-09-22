close all; clear all;

%==========================================================================
% Neural parameters and model:
%--------------------------------------------------------------------------
P.K = 4;
P.M = 2*P.K;

A = [-3.0  -1.5  ; 
      0.2  -0.2 ];
ACell = repmat({A}, 1, P.K);
P.A = blkdiag(ACell{:});

P.C = 1;

P.T  = 30;
P.dt = 0.01;

U = zeros(P.M, P.T/P.dt);
dur    = 2/P.dt;
onset  = 3/P.dt;
offset = onset + dur;
U(1,onset:offset) = 0.25;
U(3,onset:offset) = 0.50;
U(5,onset:offset) = 0.75;
U(7,onset:offset) = 1.00;

X = DCM_model(P, U);

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
  
% Specify LBR model:
%--------------------------------------------------------------------------  
P.H       = LBR_parameters(P.K); % get default parameters (see inside the function), 
                               % NOTE: By default baseline CBV is increasing towards the surface in the ascending vein
P.H.T     = P.T;               % copy the lenght of the response from neuronal specification
  
P.H.alpha_v   = 0.35;          % Choose steady-state CBF-CBV coupling for venules
P.H.alpha_d   = 0.2;           % Choose steady-state CBF-CBV coupling for ascending vein
P.H.tau_d_de  = 30;            % Choose dynamic CBF-CBV uncoupling for ascending vein

[LBR,Y] = LBR_model(P.H,cbf);  % Generate the laminar bold response


time_axis = [0:P.H.dt:P.H.T-P.H.dt] - onset*P.dt; % time axis in seconds

% Display underlying physiological responses
figure(2),
subplot(231), plot(time_axis,cbf); xlim([time_axis(1), time_axis(end)]); ylim([0.8 1.6]);
xlabel('Time (s)'); ylabel('Relative CBF in MV (%)'); axis square; 
subplot(232), plot(time_axis,Y.mv); xlim([time_axis(1), time_axis(end)]); ylim([0.8 1.6]);
xlabel('Time (s)'); ylabel('Relative CMRO_2 in MV (%)'); axis square;
subplot(233), plot(time_axis,Y.vv); xlim([time_axis(1), time_axis(end)]); ylim([0.8 1.6]);
xlabel('Time (s)'); ylabel('Relative CBV in MV (%)'); axis square;
subplot(234), plot(time_axis,Y.qv); xlim([time_axis(1), time_axis(end)]); ylim([0.7 1.2]);
xlabel('Time (s)'); ylabel('Relative dHb in MV (%)'); axis square;
subplot(235), plot(time_axis,Y.vd); xlim([time_axis(1), time_axis(end)]); ylim([0.8 1.6]);
xlabel('Time (s)'); ylabel('Relative CBV in AV (%)'); axis square;
subplot(236), p = plot(time_axis,Y.qd); xlim([time_axis(1), time_axis(end)]); ylim([0.7 1.2]);
xlabel('Time (s)'); ylabel('Relative dHb in AV (%)'); axis square; legend([p(1) p(end)],{'Upper','Lower'});
 

% Display laminar BOLD response
figure(3),
subplot(131), p = plot(time_axis,LBR); xlim([time_axis(1), time_axis(end)]); ylim([-1 4]);  %                         
xlabel('Time (s)'); ylabel('LBR (%)'); axis square;  legend([p(1) p(end)],{'Upper','Lower'});

% calculate time to peak (TTP) and time to undershoot (TTU) with respect to
% the stimulus onset and offset, respectively
[Peak_Amp,Peak_Pos] = max(LBR(onset:end,:));
[PSU_Amp,PSU_Pos]   = min(LBR(offset:end,:));
TTP = time_axis(onset+Peak_Pos)';
TTU = time_axis(offset+PSU_Pos)'-(offset-onset)*P.dt;  
% Display TTP and TTU as function of cortical depth
subplot(132), plot(P.H.l,flipud(TTP),'.-'); xlim([0 100]); ylim([0 12]);  %                          
xlabel('1 - Cortical depth (%)'); ylabel('TTP (s)'); axis square;
subplot(133), plot(P.H.l,flipud(TTU),'.-'); xlim([0 100]); ylim([0 12]);  %                         
xlabel('1 - Cortical depth (%)'); ylabel('TTU (%)'); axis square;
  