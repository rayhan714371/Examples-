% Improved Numerical Example for Section IV of 
% "Proactive attack detection scheme based on nonlinear transformation and moving target defense"

% System parameters
dt = 0.01;
gamma = 0.25;
simTime = 40; % Simulation time (includes attack interval [20,30])
numSteps = simTime / dt;

% State-space matrices (UAS dynamics) from the paper
A = [1, 0, (1 - 0.5*dt)*dt, 0;
     0, 1, 0, (1 - 0.5*gamma*dt)*dt;
     0, 0, 1 - gamma*dt, 0;
     0, 0, 0, 1 - gamma*dt];
B = [0.5*dt^2, 0;
     0, 0.5*dt^2;
     dt, 0;
     0, dt];
C = eye(4);

% Controller and observer gains
K = [40.0400, 0, 29.5498, 0;
     0, 20.2002, 0, 68.7490];
L = [0.2000, 0, 0.0499, 0;
     0, 0.2000, 0, 0.0499;
     0, 0, 0.4975, 0;
     0, 0, 0, 0.0975];

% Initial states
x = [10; -20; 30; -10]; % True state
hat_x = zeros(4,1);      % Estimated state (initialized to 0)

% Noise bounds (UBB)
delta = 0.1;  % Process noise bound
epsilon = 0.1; % Measurement noise bound

% Nonlinear transformation functions (element-wise cubic)
f = @(y) y.^3;               % y' = y^3
f_inv = @(y) sign(y).*abs(y).^(1/3); % Inverse: y = y'^(1/3)

% Moving target defense (Γ_k: time-varying diagonal matrix)
% Increased variation to make attacks more detectable
Gamma_k = @(k) diag(1 + 0.8*sin(k/10)*ones(4,1)); % Increased amplitude from 0.5 to 0.8

% Attack parameters
attack_start = 20/dt;   % Start at 20 seconds (step index)
attack_end = 30/dt;     % End at 30 seconds
% Increased attack magnitude for better detection
a_y = [15; 15; 5; 5];     % Increased attack magnitude

% Threshold for residual detection - LOWERED for better detection
threshold = 0.4; % Lowered from 0.8 to 0.4

% Storage variables
Px = zeros(1, numSteps + 1);
Py = zeros(1, numSteps + 1);
residual = zeros(4, numSteps + 1);
residual_norm = zeros(1, numSteps + 1);
flags = zeros(1, numSteps + 1);
time = 0:dt:simTime;

% Initialize
Px(1) = x(1);
Py(1) = x(2);

% Storage for estimation errors
e_Px = zeros(1, numSteps + 1);
e_Py = zeros(1, numSteps + 1);
e_Vx = zeros(1, numSteps + 1);
e_Vy = zeros(1, numSteps + 1);

% Storage for visualizing attack effects
y_original = zeros(4, numSteps);
y_bar_attacked = zeros(4, numSteps);

% Simulation loop
for k = 1:numSteps
    % Control input
    u = -K * hat_x;  % CHANGE: Use estimated state for control, not true state
    
    % Update true state with process noise (bounded)
    omega = delta * (2*rand(4,1) - 1); % Random in [-delta, delta]
    x = A * x + B * u + omega;
    
    % Measurement with noise
    v = epsilon * (2*rand(4,1) - 1);   % Random in [-epsilon, epsilon]
    y_k = C * x + v;
    
    % Store original measurement for comparison
    y_original(:, k) = y_k;
    
    % STEP 1 OF ALGORITHM 1: Calculate y'_k (nonlinear transformation)
    y_prime = f(y_k);
    
    % STEP 2 OF ALGORITHM 1: Calculate y^M_k (moving target)
    Gamma = Gamma_k(k); % Time-varying Γ
    y_M = Gamma * y_prime;
    
    % Simulate FDI attack injection
    if k >= attack_start && k <= attack_end
        y_M_attacked = y_M + a_y; % Inject attack
    else
        y_M_attacked = y_M;       % No attack
    end
    
    % STEP 3 OF ALGORITHM 1: Calculate y_bar (inverse transformations at receiver)
    y_prime_attacked = Gamma \ y_M_attacked; % Γ^{-1} * y_M_attacked
    y_bar = f_inv(y_prime_attacked);         % Recover measurement
    
    y_bar_attacked(:, k) = y_bar; % Store for visualization
    
    % STEP 4 OF ALGORITHM 1: State estimation
    hat_x_pred = A * hat_x + B * u;  % Predicted state
    
    % STEP 5 OF ALGORITHM 1: Residual calculation r^a_k = y_bar - C*x_hat_pred
    residual(:, k+1) = y_bar - C * hat_x_pred;
    residual_norm(k+1) = norm(residual(:, k+1));
    
    % Update state estimate
    hat_x = hat_x_pred + L * residual(:, k+1);
    
    % Attack detection
    if residual_norm(k+1) > threshold
        flags(k+1) = 1; % Attack detected
    else
        flags(k+1) = 0; % No attack
    end
    
    % Calculate estimation error
    e = x - hat_x;
    e_Px(k+1) = e(1);
    e_Py(k+1) = e(2);
    e_Vx(k+1) = e(3);
    e_Vy(k+1) = e(4);
    
    % Store trajectory
    Px(k+1) = x(1);
    Py(k+1) = x(2);
end

% Create figures similar to the paper
% Figure 1: State Estimation Errors
figure;
subplot(4,1,1);
plot(time, e_Px, 'b', 'LineWidth', 1.5);
hold on;
xline(20, '--r', 'Attack Start');
xline(30, '--r', 'Attack End');
ylabel('P_x Error (m)');
title('State Estimation Errors');
grid on;

subplot(4,1,2);
plot(time, e_Py, 'b', 'LineWidth', 1.5);
hold on;
xline(20, '--r', 'Attack Start');
xline(30, '--r', 'Attack End');
ylabel('P_y Error (m)');
grid on;

subplot(4,1,3);
plot(time, e_Vx, 'b', 'LineWidth', 1.5);
hold on;
xline(20, '--r', 'Attack Start');
xline(30, '--r', 'Attack End');
ylabel('V_x Error (m/s)');
grid on;

subplot(4,1,4);
plot(time, e_Vy, 'b', 'LineWidth', 1.5);
hold on;
xline(20, '--r', 'Attack Start');
xline(30, '--r', 'Attack End');
xlabel('Time (s)');
ylabel('V_y Error (m/s)');
grid on;

% Figure 2: UAS Trajectory and Attack Detection
figure;
subplot(2,1,1);
plot(Px, Py, 'b-', 'LineWidth', 1.5);
hold on;
scatter(Px(attack_start:attack_end), Py(attack_start:attack_end), 10, 'r', 'filled');
xlabel('P_x (m)');
ylabel('P_y (m)');
title('UAS Trajectory (Red: Attack Period)');
grid on;

subplot(2,1,2);
stairs(time, flags, 'r-', 'LineWidth', 1.5);
hold on;
plot(time, residual_norm, 'b-', 'LineWidth', 1);
yline(threshold, '--g', 'Threshold');
xlabel('Time (s)');
ylabel('Attack Flag / Residual Norm');
title('Attack Detection Results');
ylim([-0.1 max(residual_norm)*1.1]);
xlim([0 simTime]);
legend('Attack Flag', 'Residual Norm', 'Threshold');
grid on;

% Figure 3: Compare original and attacked measurements (first coordinate)
figure;
subplot(2,1,1);
plot(dt:dt:simTime, y_original(1,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(dt:dt:simTime, y_bar_attacked(1,:), 'r--', 'LineWidth', 1.5);
xline(20, '--k', 'Attack Start');
xline(30, '--k', 'Attack End');
xlabel('Time (s)');
ylabel('Position X');
title('Effect of Attack on X-Position Measurement');
legend('Original Measurement', 'Received Measurement');
grid on;

subplot(2,1,2);
plot(dt:dt:simTime, y_original(2,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(dt:dt:simTime, y_bar_attacked(2,:), 'r--', 'LineWidth', 1.5);
xline(20, '--k', 'Attack Start');
xline(30, '--k', 'Attack End');
xlabel('Time (s)');
ylabel('Position Y');
title('Effect of Attack on Y-Position Measurement');
legend('Original Measurement', 'Received Measurement');
grid on;

% Additional figure to visualize residuals during attack
figure;
subplot(2,1,1);
plot(time, residual_norm, 'b-', 'LineWidth', 1.5);
hold on;
plot([0 simTime], [threshold threshold], 'r--', 'LineWidth', 1.5);
xline(20, '--k', 'Attack Start');
xline(30, '--k', 'Attack End');
xlabel('Time (s)');
ylabel('Residual Norm');
title('Residual Norm with Detection Threshold');
legend('Residual Norm', 'Threshold');
grid on;

subplot(2,1,2);
bar(time, flags, 'r');
xlabel('Time (s)');
ylabel('Attack Detection Flag');
title('Attack Detection Result (1=Attack Detected)');
ylim([-0.1 1.1]);
grid on;

% Calculate detection performance metrics
attack_period = attack_start:attack_end;
true_positives = sum(flags(attack_period) == 1);
false_negatives = sum(flags(attack_period) == 0);
non_attack_period = [1:attack_start-1, attack_end+1:numSteps+1];
false_positives = sum(flags(non_attack_period) == 1);
true_negatives = sum(flags(non_attack_period) == 0);

detection_rate = true_positives / length(attack_period) * 100;
false_alarm_rate = false_positives / length(non_attack_period) * 100;

fprintf('Detection Performance Metrics:\n');
fprintf('Detection Rate: %.2f%%\n', detection_rate);
fprintf('False Alarm Rate: %.2f%%\n', false_alarm_rate);
fprintf('Number of True Positives: %d\n', true_positives);
fprintf('Number of False Negatives: %d\n', false_negatives);
fprintf('Number of False Positives: %d\n', false_positives);
fprintf('Number of True Negatives: %d\n', true_negatives);