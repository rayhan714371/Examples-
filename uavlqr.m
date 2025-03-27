%% Proactive Attack Detection Scheme for Cyber-Physical Systems
% This code implements the numerical example from the paper

% Clear workspace
clear all;
clc;
close all;

%% System Parameters
dt = 0.01;      % Sampling time
gamma = 0.25;   % Damping parameter
T = 50;         % Simulation time
N = T/dt;       % Number of time steps

% State-space matrices (UAS dynamics)
A = [1, 0, (1 - 0.5*dt)*dt, 0;
     0, 1, 0, (1 - 0.5*gamma*dt)*dt;
     0, 0, 1 - gamma*dt, 0;
     0, 0, 0, 1 - gamma*dt];
 
B = [0.5*dt^2, 0;
     0, 0.5*dt^2;
     dt, 0;
     0, dt];
 
C = eye(4);     % Output matrix
 
% Controller and observer gains (given)
K = [40.0400, 0, 29.5498, 0;
     0, 20.2002, 0, 68.7490];
 
L = [0.2000, 0, 0.0499, 0;
     0, 0.2000, 0, 0.0499;
     0, 0, 0.4975, 0;
     0, 0, 0, 0.0975];

% Initial state and bounds
x_0 = [10; -20; 30; -10];    % Initial state [P_x; P_y; V_x; V_y]
x_hat_0 = zeros(4,1);         % Initial state estimate
e_0 = x_0 - x_hat_0;          % Initial estimation error

% Noise bounds
delta = 0.01;   % Process noise bound
epsilon = 0.01; % Measurement noise bound

%% Nonlinear Transformation and MTD Parameters
% Using a hyperbolic tangent function for nonlinear transformation
% f(x) = tanh(alpha*x)
alpha = 0.1;  % Parameter for nonlinear transformation (reduced to avoid extreme values)

% Define the nonlinear transformation function and its inverse
f = @(x) tanh(alpha*x);
f_inv = @(x) atanh(max(min(x, 0.99), -0.99))/alpha; % The inverse of tanh is arctanh

% Moving Target Defense parameters
% Time-varying diagonal matrix Γ_k
% Using values between 1.5 and 2.0 for more stable computation
vartheta_min = 1.5;
vartheta_max = 2.0;

%% Attack Parameters
attack_start = 20/dt;     % Attack starts at t = 20s
attack_end = 30/dt;       % Attack ends at t = 30s
attack_magnitude = 0.5;   % Reduced attack magnitude for stability

% Detection threshold calculation
% Based on expected noise levels and system dynamics
r_threshold = 5*epsilon;  % Increased threshold for better detection accuracy

%% Initialize arrays for storage
x = zeros(4, N+1);        % System states
x_hat = zeros(4, N+1);    % State estimates
y = zeros(4, N+1);        % Measurements
y_prime = zeros(4, N+1);  % Transformed measurements
yM = zeros(4, N+1);       % Moving target outputs
y_bar = zeros(4, N+1);    % Restored measurements
r_a = zeros(4, N+1);      % Residuals
e = zeros(4, N+1);        % Estimation errors
attack_flag = zeros(1, N+1); % Attack detection flag
true_attack_status = zeros(1, N+1); % Ground truth for attack presence
Gamma_k_values = zeros(4, N+1); % Store Gamma_k diagonal values
attacks = zeros(4, N+1);   % Store attacks applied

% Set initial values
x(:,1) = x_0;
x_hat(:,1) = x_hat_0;
e(:,1) = e_0;

%% Control Input (for trajectory to converge to origin)
u = zeros(2, N);

%% Main Simulation Loop
for k = 1:N
    % Generate bounded noise
    omega_k = (2*rand(4,1)-1) * delta;    % Process noise
    v_k = (2*rand(4,1)-1) * epsilon;      % Measurement noise
    
    % Compute control input to drive system to origin
    u(:,k) = -K * x(:,k);
    
    % True system state update with control input
    x(:,k+1) = A * x(:,k) + B * u(:,k) + omega_k;
    
    % Measurement from sensors
    y(:,k) = C * x(:,k) + v_k;
    
    % Create the Gamma_k matrix for this time step
    Gamma_k_values(:,k) = vartheta_min + (vartheta_max - vartheta_min) * rand(4,1);
    Gamma_k = diag(Gamma_k_values(:,k));
    
    % Apply nonlinear transformation
    y_prime(:,k) = f(y(:,k));
    
    % Apply moving target defense
    yM(:,k) = Gamma_k * y_prime(:,k);
    
    % Apply FDI attack if in attack interval
    attacks(:,k) = zeros(4,1);
    if k >= attack_start && k <= attack_end
        % Attack model: additive attack on the transformed measurements
        attacks(:,k) = attack_magnitude * ones(4,1);
        true_attack_status(k) = 1; % Ground truth - attack is present
    else
        true_attack_status(k) = 0; % Ground truth - no attack
    end
    ya_k = yM(:,k) + attacks(:,k);
    
    % Restore original measurement (attack will be transformed too)
    y_tilda_k = Gamma_k \ ya_k;
    
    % Apply inverse transformation with saturation to avoid numerical issues
    y_bar(:,k) = f_inv(y_tilda_k);
    
    % State estimation
    x_hat(:,k+1) = A * x_hat(:,k) + B * u(:,k) + L * (y_bar(:,k) - C * x_hat(:,k));
    
    % Compute estimation error
    e(:,k+1) = x(:,k+1) - x_hat(:,k+1);
    
    % Compute residual for attack detection
    r_a(:,k) = y_bar(:,k) - C * x_hat(:,k);
    
    % Attack detection logic
    if norm(r_a(:,k)) > r_threshold
        attack_flag(k) = 1;  % Attack detected
    else
        attack_flag(k) = 0;  % No attack
    end
end

%% Calculate Detection Performance Metrics
% Note: We exclude k=N+1 since there's no detection for it
true_attack_status = true_attack_status(1:N); % Ground truth for attack presence
attack_flag = attack_flag(1:N); % Detection results

% True Positives: Attack present and detected
TP = sum(true_attack_status == 1 & attack_flag == 1);

% False Negatives: Attack present but not detected
FN = sum(true_attack_status == 1 & attack_flag == 0);

% False Positives: No attack but detection triggered
FP = sum(true_attack_status == 0 & attack_flag == 1);

% True Negatives: No attack and no detection
TN = sum(true_attack_status == 0 & attack_flag == 0);

% Detection Rate (True Positive Rate, Sensitivity, Recall) in percentage
DR_pct = 100 * TP / (TP + FN);

% False Alarm Rate (False Positive Rate) in percentage
FAR_pct = 100 * FP / (FP + TN);

% Display metrics
fprintf('\n===== Detection Performance Metrics =====\n');
fprintf('Detection Rate: %.2f%%\n', DR_pct);
fprintf('False Alarm Rate: %.2f%%\n', FAR_pct);
fprintf('Number of True Positives: %d\n', TP);
fprintf('Number of False Negatives: %d\n', FN);
fprintf('Number of False Positives: %d\n', FP);
fprintf('Number of True Negatives: %d\n', TN);
fprintf('=======================================\n\n');

%% Create confusion matrix visualization
figure;
cm = confusionchart([TN, FP; FN, TP], {'No Attack', 'Attack'});
cm.Title = 'Attack Detection Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% Create a text box with metrics on the confusion matrix
annotation('textbox', [0.15, 0.05, 0.7, 0.1], ...
    'String', {sprintf('Detection Rate: %.2f%%', DR_pct), ...
               sprintf('False Alarm Rate: %.2f%%', FAR_pct)}, ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 12, ...
    'FontWeight', 'bold');

%% Plotting Results
time = (0:N)*dt;

% Figure 1: Trajectory of P_x and P_y (entire trajectory)
figure;
plot(x(1,:), x(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(x(1,1), x(2,1), 'go', 'MarkerSize', 10, 'LineWidth', 2);  % Initial point
plot(0, 0, 'kx', 'MarkerSize', 10, 'LineWidth', 2);  % Target point

% Mark attack period with different color
attack_indices = attack_start:attack_end;
plot(x(1,attack_indices), x(2,attack_indices), 'r-', 'LineWidth', 2.5);

title('Figure 1: Trajectory of P_x and P_y');
xlabel('P_x');
ylabel('P_y');
legend('Trajectory', 'Initial Position', 'Target Position', 'Under Attack');
grid on;

% Figure 2: State estimation error (attack-free)
figure;
subplot(4,1,1);
plot(time(1:attack_start), e(1,1:attack_start), 'b-', 'LineWidth', 1.5);
title('Figure 2: State Estimation Error (Attack-free)');
ylabel('e_{P_x}');
grid on;

subplot(4,1,2);
plot(time(1:attack_start), e(2,1:attack_start), 'b-', 'LineWidth', 1.5);
ylabel('e_{P_y}');
grid on;

subplot(4,1,3);
plot(time(1:attack_start), e(3,1:attack_start), 'b-', 'LineWidth', 1.5);
ylabel('e_{V_x}');
grid on;

subplot(4,1,4);
plot(time(1:attack_start), e(4,1:attack_start), 'b-', 'LineWidth', 1.5);
ylabel('e_{V_y}');
xlabel('Time (s)');
grid on;

% Figure 3: Trajectory change under attack (position vs. time)
figure;
subplot(2,1,1);
plot(time, x(1,:), 'b-', 'LineWidth', 1.5);
title('Figure 3: Trajectory Change Under Attack');
ylabel('P_x');
grid on;

% Add shaded region for attack period
attack_region_x = [time(attack_start), time(attack_end), time(attack_end), time(attack_start)];
attack_region_y = [min(x(1,:))-1, min(x(1,:))-1, max(x(1,:))+1, max(x(1,:))+1];
hold on;
patch(attack_region_x, attack_region_y, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

subplot(2,1,2);
plot(time, x(2,:), 'r-', 'LineWidth', 1.5);
ylabel('P_y');
xlabel('Time (s)');
grid on;

% Add shaded region for attack period
attack_region_x = [time(attack_start), time(attack_end), time(attack_end), time(attack_start)];
attack_region_y = [min(x(2,:))-1, min(x(2,:))-1, max(x(2,:))+1, max(x(2,:))+1];
hold on;
patch(attack_region_x, attack_region_y, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
text((time(attack_start) + time(attack_end))/2, min(x(2,:))-0.5, 'Attack Period', 'HorizontalAlignment', 'center');

% Figure 4: Estimation error during attack
figure;
subplot(4,1,1);
plot(time, e(1,:), 'b-', 'LineWidth', 1.5);
title('Figure 4: Estimation Error During Attack');
ylabel('e_{P_x}');
grid on;

subplot(4,1,2);
plot(time, e(2,:), 'b-', 'LineWidth', 1.5);
ylabel('e_{P_y}');
grid on;

subplot(4,1,3);
plot(time, e(3,:), 'b-', 'LineWidth', 1.5);
ylabel('e_{V_x}');
grid on;

subplot(4,1,4);
plot(time, e(4,:), 'b-', 'LineWidth', 1.5);
ylabel('e_{V_y}');
xlabel('Time (s)');
grid on;

% Add shaded regions for attack period across all subplots
for i = 1:4
    subplot(4,1,i);
    attack_region_x = [time(attack_start), time(attack_end), time(attack_end), time(attack_start)];
    attack_region_y = [min(e(i,:))-0.5, min(e(i,:))-0.5, max(e(i,:))+0.5, max(e(i,:))+0.5];
    hold on;
    patch(attack_region_x, attack_region_y, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end

% Figure 5: Attack detection results with metrics
figure;
subplot(2,1,1);
stem(time(1:N), true_attack_status, 'filled', 'LineWidth', 1.5);
title(sprintf('Attack Detection Results (DR: %.2f%%, FAR: %.2f%%)', DR_pct, FAR_pct));
ylabel('True Attack Status');
ylim([-0.1, 1.1]);
yticks([0 1]);
yticklabels({'No Attack', 'Attack Present'});
grid on;

subplot(2,1,2);
stem(time(1:N), attack_flag, 'filled', 'LineWidth', 1.5);
ylabel('Attack Flag');
xlabel('Time (s)');
ylim([-0.1, 1.1]);
yticks([0 1]);
yticklabels({'No Attack', 'Attack Detected'});
grid on;

% Add shaded region for attack period
hold on;
attack_region_x = [time(attack_start), time(attack_end), time(attack_end), time(attack_start)];
attack_region_y = [-0.1, -0.1, 1.1, 1.1];
patch(attack_region_x, attack_region_y, 'red', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
text((time(attack_start) + time(attack_end))/2, -0.05, 'Actual Attack Period', 'HorizontalAlignment', 'center');

% Figure 6: Residual magnitude compared to threshold
figure;
plot(time(1:N), arrayfun(@(k) norm(r_a(:,k)), 1:N), 'b-', 'LineWidth', 1.5);
hold on;
plot(time(1:N), r_threshold*ones(1,N), 'r--', 'LineWidth', 1.5);
title('Figure 6: Residual Magnitude vs. Detection Threshold');
xlabel('Time (s)');
ylabel('||r_a||');
legend('Residual Magnitude', 'Detection Threshold');
grid on;

% Add shaded region for attack period
attack_region_x = [time(attack_start), time(attack_end), time(attack_end), time(attack_start)];
attack_region_y = [0, 0, max(arrayfun(@(k) norm(r_a(:,k)), 1:N))*1.1, max(arrayfun(@(k) norm(r_a(:,k)), 1:N))*1.1];
hold on;
patch(attack_region_x, attack_region_y, 'red', 'FaceAlpha', 0.1, 'EdgeColor', 'none');