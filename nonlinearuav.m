% 6-DOF UAV Model with Moving Target Defense for Attack Detection
% Implementation based on linearization and Moving Target Defense (MTD)
% Comparing effectiveness with and without linearization error

% System parameters
dt = 0.01;
simTime = 40;    % Simulation time (includes attack interval [20,30])
numSteps = simTime / dt;

% ============= 6-DOF UAV Nonlinear Model and Linearization =============
% Define the nonlinear 6-DOF UAV model states:
% x = [x, y, z, phi, theta, psi, u, v, w, p, q, r]'
% where (x,y,z) position, (phi,theta,psi) are Euler angles,
% (u,v,w) are body-fixed velocities, and (p,q,r) are angular rates

% Constants
g = 9.81;        % Gravity (m/s^2)
m = 1.5;         % UAV mass (kg)
Ix = 0.1;        % Moment of inertia around x-axis (kg·m^2)
Iy = 0.12;       % Moment of inertia around y-axis (kg·m^2)
Iz = 0.15;       % Moment of inertia around z-axis (kg·m^2)

% Linearized state-space model around hover
% State vector: x = [x, y, z, phi, theta, psi, u, v, w, p, q, r]'
% Control input: u = [F, tau_x, tau_y, tau_z]' (thrust and moments)

% Linearized A matrix
% Simplified linearization for hover condition
A_c = zeros(12, 12);
% Position dynamics
A_c(1,7) = 1;  % dx/dt = u
A_c(2,8) = 1;  % dy/dt = v
A_c(3,9) = 1;  % dz/dt = w
% Rotational kinematics
A_c(4,10) = 1; % dphi/dt = p
A_c(5,11) = 1; % dtheta/dt = q
A_c(6,12) = 1; % dpsi/dt = r
% Linear velocity dynamics (includes gravity effects)
A_c(7,5) = -g; % du/dt affected by pitch (theta)
A_c(8,4) = g;  % dv/dt affected by roll (phi)
% Angular velocity dynamics - simplified
A_c(10,10) = -0.1; % damping in p
A_c(11,11) = -0.1; % damping in q
A_c(12,12) = -0.1; % damping in r

% Linearized B matrix
B_c = zeros(12, 4);
% Control inputs affect accelerations
B_c(9,1) = 1/m;      % Thrust affects w-acceleration
B_c(10,2) = 1/Ix;    % Roll moment affects p-acceleration
B_c(11,3) = 1/Iy;    % Pitch moment affects q-acceleration
B_c(12,4) = 1/Iz;    % Yaw moment affects r-acceleration

% Discretize the continuous system
[A, B] = c2d(A_c, B_c, dt);

% Output matrix - we observe all states
C = eye(12);

% Initial conditions
x_true = zeros(12, 1);
x_true(1:3) = [10; -5; -2];  % Initial position
x_hat = zeros(12, 1);        % Initial state estimate

% ================ LQR Controller Design ================
% Design LQR controller for the linearized system
Q = diag([10, 10, 10, 5, 5, 5, 1, 1, 1, 0.1, 0.1, 0.1]);
R = diag([0.1, 0.1, 0.1, 0.1]);
[K_lqr, ~, ~] = dlqr(A, B, Q, R);

% ================ Kalman Filter Design ================
% Process and measurement noise covariances
Q_kf = 0.01 * eye(12);
R_kf = 0.01 * eye(12);
% Initial error covariance
P = 10 * eye(12);
% Calculate Kalman filter gain offline (time-invariant)
[~, ~, L] = dlqe(A, eye(12), C, Q_kf, R_kf);

% ================ Moving Target Defense Setup ================
% Noise bounds (UBB)
delta = 0.1;     % Process noise bound
epsilon = 0.1;   % Measurement noise bound

% Moving target defense (Γ_k: time-varying diagonal matrix)
% Generate a time-varying diagonal matrix for MTD
Gamma_k = @(k) diag(1 + 0.5*sin(k/10)*ones(12,1));

% Attack parameters
attack_start = 20/dt;   % Start at 20 seconds (step index)
attack_end = 30/dt;     % End at 30 seconds
% Attack vector (higher values for position states)
a_y = [10; 10; 10; 1; 1; 1; 2; 2; 2; 0.5; 0.5; 0.5];

% Detection threshold
threshold = 2.0;

% ================ Storage Variables ================
% For trajectory and analysis
pos_x = zeros(1, numSteps + 1);
pos_y = zeros(1, numSteps + 1);
pos_z = zeros(1, numSteps + 1);

% For estimation errors
state_errors = zeros(12, numSteps + 1);
error_norms = zeros(1, numSteps + 1);

% For attack detection
residual = zeros(12, numSteps + 1);
residual_norm = zeros(1, numSteps + 1);
residual_norm_no_mtd = zeros(1, numSteps + 1);
flags_mtd = zeros(1, numSteps + 1);
flags_no_mtd = zeros(1, numSteps + 1);

% For comparing performance with linearization error
residual_norm_with_h = zeros(1, numSteps + 1);
residual_norm_without_h = zeros(1, numSteps + 1);
flags_with_h = zeros(1, numSteps + 1);
flags_without_h = zeros(1, numSteps + 1);

% For original and attacked measurements
y_original = zeros(12, numSteps);
y_mtd_received = zeros(12, numSteps);
y_no_mtd_received = zeros(12, numSteps);

% Initialize trajectory
pos_x(1) = x_true(1);
pos_y(1) = x_true(2);
pos_z(1) = x_true(3);

% Time vector
time = 0:dt:simTime;

% ==================== Simulation Loop ====================
for k = 1:numSteps
    % ---- Control Input ----
    u = -K_lqr * x_hat;  % LQR control using estimated state
    
    % ---- Nonlinear Model Simulation ----
    % Generate a nonlinear term h(x,u) to represent linearization error
    % This includes aerodynamic effects, coupling, etc.
    h_xu = 0.5 * sin(0.1*k) * (0.2*randn(12,1) + 0.1*sin(0.2*x_true));
    
    % Process noise (bounded)
    omega = delta * (2*rand(12,1) - 1);  % Uniform in [-delta, delta]
    
    % Update true state with full nonlinear dynamics
    x_true = A * x_true + B * u + h_xu + omega;
    
    % ---- Measurement ----
    % Measurement noise (bounded)
    v = epsilon * (2*rand(12,1) - 1);  % Uniform in [-epsilon, epsilon]
    y_k = C * x_true + v;  % Measurement
    
    % Store original measurement
    y_original(:, k) = y_k;
    
    % =============== WITH MTD - WITH LINEARIZATION ERROR ===============
    % ----- Case 1: With MTD, with linearization error h(x,u) -----
    
    % Apply Moving Target Defense
    Gamma = Gamma_k(k);  % Time-varying transformation
    y_M = Gamma * y_k;   % Apply MTD transformation
    
    % Simulate FDI attack injection
    if k >= attack_start && k <= attack_end
        y_M_attacked = y_M + a_y;  % Inject attack
    else
        y_M_attacked = y_M;        % No attack
    end
    
    % Recover measurement at receiver
    y_bar = Gamma \ y_M_attacked;  % Inverse transformation
    y_mtd_received(:, k) = y_bar;  % Store for visualization
    
    % State estimation with linearization error
    x_hat_pred = A * x_hat + B * u;  % Predicted state (no h(x,u) in the model)
    
    % Residual calculation
    residual(:, k+1) = y_bar - C * x_hat_pred;
    residual_norm_with_h(k+1) = norm(residual(:, k+1));
    
    % Attack detection with MTD + linearization error
    if residual_norm_with_h(k+1) > threshold
        flags_with_h(k+1) = 1;  % Attack detected
    else
        flags_with_h(k+1) = 0;  % No attack
    end
    
    % Update state estimate
    x_hat = x_hat_pred + L * residual(:, k+1);
    
    % Store estimation error
    state_errors(:, k+1) = x_true - x_hat;
    error_norms(k+1) = norm(state_errors(:, k+1));
    
    % =============== WITH MTD - WITHOUT LINEARIZATION ERROR ===============
    % ----- Case 2: With MTD, without linearization error h(x,u) -----
    
    % In this case, we assume perfect knowledge of the linearization error
    % So we "remove" it from the dynamics by adding it to the prediction
    x_hat_pred_perfect = A * x_hat + B * u + h_xu;  % Added h(x,u)
    
    % Residual calculation with perfect model
    residual_perfect = y_bar - C * x_hat_pred_perfect;
    residual_norm_without_h(k+1) = norm(residual_perfect);
    
    % Attack detection with MTD + perfect model
    if residual_norm_without_h(k+1) > threshold
        flags_without_h(k+1) = 1;  % Attack detected
    else
        flags_without_h(k+1) = 0;  % No attack
    end
    
    % =============== WITHOUT MTD - WITH LINEARIZATION ERROR ===============
    % ----- Case 3: Without MTD, with linearization error h(x,u) -----
    
    % Without MTD, the attack directly affects measurements
    if k >= attack_start && k <= attack_end
        y_direct_attacked = y_k + a_y;  % Direct attack on measurements
    else
        y_direct_attacked = y_k;        % No attack
    end
    
    y_no_mtd_received(:, k) = y_direct_attacked;
    
    % Residual calculation without MTD
    residual_no_mtd = y_direct_attacked - C * x_hat_pred;
    residual_norm_no_mtd(k+1) = norm(residual_no_mtd);
    
    % Attack detection without MTD
    if residual_norm_no_mtd(k+1) > threshold
        flags_no_mtd(k+1) = 1;  % Attack detected
    else
        flags_no_mtd(k+1) = 0;  % No attack
    end
    
    % Store trajectory
    pos_x(k+1) = x_true(1);
    pos_y(k+1) = x_true(2);
    pos_z(k+1) = x_true(3);
end

% ===================== Performance Evaluation =====================
% Attack period and non-attack period
attack_period = attack_start:attack_end;
non_attack_period = [1:attack_start-1, attack_end+1:numSteps+1];

% ---- Performance metrics for MTD with linearization error ----
true_positives_with_h = sum(flags_with_h(attack_period) == 1);
false_negatives_with_h = sum(flags_with_h(attack_period) == 0);
false_positives_with_h = sum(flags_with_h(non_attack_period) == 1);
true_negatives_with_h = sum(flags_with_h(non_attack_period) == 0);

detection_rate_with_h = true_positives_with_h / length(attack_period) * 100;
false_alarm_rate_with_h = false_positives_with_h / length(non_attack_period) * 100;

% ---- Performance metrics for MTD without linearization error ----
true_positives_without_h = sum(flags_without_h(attack_period) == 1);
false_negatives_without_h = sum(flags_without_h(attack_period) == 0);
false_positives_without_h = sum(flags_without_h(non_attack_period) == 1);
true_negatives_without_h = sum(flags_without_h(non_attack_period) == 0);

detection_rate_without_h = true_positives_without_h / length(attack_period) * 100;
false_alarm_rate_without_h = false_positives_without_h / length(non_attack_period) * 100;

% ---- Performance metrics for No MTD (direct attack) ----
true_positives_no_mtd = sum(flags_no_mtd(attack_period) == 1);
false_negatives_no_mtd = sum(flags_no_mtd(attack_period) == 0);
false_positives_no_mtd = sum(flags_no_mtd(non_attack_period) == 1);
true_negatives_no_mtd = sum(flags_no_mtd(non_attack_period) == 0);

detection_rate_no_mtd = true_positives_no_mtd / length(attack_period) * 100;
false_alarm_rate_no_mtd = false_positives_no_mtd / length(non_attack_period) * 100;

% ===================== Visualization =====================
% Figure 1: 3D Trajectory
figure;
plot3(pos_x, pos_y, pos_z, 'b-', 'LineWidth', 1.5);
hold on;
plot3(pos_x(attack_start:attack_end), pos_y(attack_start:attack_end), pos_z(attack_start:attack_end), 'r-', 'LineWidth', 2);
plot3(pos_x(1), pos_y(1), pos_z(1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(pos_x(end), pos_y(end), pos_z(end), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
grid on;
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
title('6-DOF UAV Trajectory (Red: Attack Period)');
legend('Normal Flight', 'Under Attack', 'Start', 'End');

% Figure 2: State Estimation Errors (Position and Orientation)
figure;
subplot(3,2,1);
plot(time, state_errors(1,:), 'b-', 'LineWidth', 1.5);
hold on;
xline(20, '--r', 'Attack Start');
xline(30, '--r', 'Attack End');
ylabel('X Position Error (m)');
title('Position Errors');
grid on;

subplot(3,2,3);
plot(time, state_errors(2,:), 'b-', 'LineWidth', 1.5);
hold on;
xline(20, '--r');
xline(30, '--r');
ylabel('Y Position Error (m)');
grid on;

subplot(3,2,5);
plot(time, state_errors(3,:), 'b-', 'LineWidth', 1.5);
hold on;
xline(20, '--r');
xline(30, '--r');
xlabel('Time (s)');
ylabel('Z Position Error (m)');
grid on;

subplot(3,2,2);
plot(time, state_errors(4,:), 'b-', 'LineWidth', 1.5);
hold on;
xline(20, '--r', 'Attack Start');
xline(30, '--r', 'Attack End');
ylabel('Roll Error (rad)');
title('Orientation Errors');
grid on;

subplot(3,2,4);
plot(time, state_errors(5,:), 'b-', 'LineWidth', 1.5);
hold on;
xline(20, '--r');
xline(30, '--r');
ylabel('Pitch Error (rad)');
grid on;

subplot(3,2,6);
plot(time, state_errors(6,:), 'b-', 'LineWidth', 1.5);
hold on;
xline(20, '--r');
xline(30, '--r');
xlabel('Time (s)');
ylabel('Yaw Error (rad)');
grid on;

% Figure 3: Comparison of Residual Norms
figure;
subplot(2,1,1);
plot(time, residual_norm_with_h, 'b-', 'LineWidth', 1.5);
hold on;
plot(time, residual_norm_without_h, 'g-', 'LineWidth', 1.5);
plot(time, residual_norm_no_mtd, 'r-', 'LineWidth', 1.5);
plot([0 simTime], [threshold threshold], 'k--', 'LineWidth', 1.5);
xline(20, '--k', 'Attack Start');
xline(30, '--k', 'Attack End');
title('Residual Norm Comparison');
ylabel('Residual Norm');
grid on;
legend('MTD with h(x,u)', 'MTD without h(x,u)', 'No MTD', 'Threshold');

subplot(2,1,2);
stairs(time, flags_with_h, 'b-', 'LineWidth', 1.5);
hold on;
stairs(time, flags_without_h, 'g-', 'LineWidth', 1.5);
stairs(time, flags_no_mtd, 'r-', 'LineWidth', 1.5);
xline(20, '--k', 'Attack Start');
xline(30, '--k', 'Attack End');
xlabel('Time (s)');
ylabel('Attack Flag');
title('Attack Detection Results');
ylim([-0.1 1.1]);
grid on;
legend('MTD with h(x,u)', 'MTD without h(x,u)', 'No MTD');

% Figure 4: X-Position Measurement Comparison
figure;
subplot(3,1,1);
plot(dt:dt:simTime, y_original(1,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(dt:dt:simTime, y_mtd_received(1,:), 'r--', 'LineWidth', 1.5);
xline(20, '--k', 'Attack Start');
xline(30, '--k', 'Attack End');
ylabel('X Position');
title('X-Position Measurement (With MTD)');
legend('Original', 'With MTD');
grid on;

subplot(3,1,2);
plot(dt:dt:simTime, y_original(1,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(dt:dt:simTime, y_no_mtd_received(1,:), 'r--', 'LineWidth', 1.5);
xline(20, '--k', 'Attack Start');
xline(30, '--k', 'Attack End');
ylabel('X Position');
title('X-Position Measurement (Without MTD)');
legend('Original', 'Without MTD');
grid on;

subplot(3,1,3);
% Plot the difference between measurements
plot(dt:dt:simTime, abs(y_mtd_received(1,:) - y_original(1,:)), 'g-', 'LineWidth', 1.5);
hold on;
plot(dt:dt:simTime, abs(y_no_mtd_received(1,:) - y_original(1,:)), 'r-', 'LineWidth', 1.5);
xline(20, '--k', 'Attack Start');
xline(30, '--k', 'Attack End');
xlabel('Time (s)');
ylabel('|Measurement Error|');
title('Absolute Measurement Error in X-Position');
legend('With MTD', 'Without MTD');
grid on;

% Print performance metrics
fprintf('\n========== Performance Metrics ==========\n');
fprintf('------------ MTD with linearization error h(x,u) ------------\n');
fprintf('Detection Rate: %.2f%%\n', detection_rate_with_h);
fprintf('False Alarm Rate: %.2f%%\n', false_alarm_rate_with_h);
fprintf('True Positives: %d\n', true_positives_with_h);
fprintf('False Negatives: %d\n', false_negatives_with_h);
fprintf('False Positives: %d\n', false_positives_with_h);
fprintf('True Negatives: %d\n', true_negatives_with_h);

fprintf('\n------------ MTD without linearization error h(x,u) ------------\n');
fprintf('Detection Rate: %.2f%%\n', detection_rate_without_h);
fprintf('False Alarm Rate: %.2f%%\n', false_alarm_rate_without_h);
fprintf('True Positives: %d\n', true_positives_without_h);
fprintf('False Negatives: %d\n', false_negatives_without_h);
fprintf('False Positives: %d\n', false_positives_without_h);
fprintf('True Negatives: %d\n', true_negatives_without_h);

fprintf('\n------------ Without MTD (Direct Attack) ------------\n');
fprintf('Detection Rate: %.2f%%\n', detection_rate_no_mtd);
fprintf('False Alarm Rate: %.2f%%\n', false_alarm_rate_no_mtd);
fprintf('True Positives: %d\n', true_positives_no_mtd);
fprintf('False Negatives: %d\n', false_negatives_no_mtd);
fprintf('False Positives: %d\n', false_positives_no_mtd);
fprintf('True Negatives: %d\n', true_negatives_no_mtd);

% Effectiveness comparison
fprintf('\n========== Effectiveness Comparison ==========\n');
fprintf('Scenario 1: MTD with linearization error\n');
fprintf('Scenario 2: MTD without linearization error\n');
fprintf('Scenario 3: No MTD (direct attack)\n\n');

% Compare detection rates
fprintf('Detection Rate Comparison:\n');
fprintf('Scenario 1: %.2f%%\n', detection_rate_with_h);
fprintf('Scenario 2: %.2f%%\n', detection_rate_without_h);
fprintf('Scenario 3: %.2f%%\n', detection_rate_no_mtd);

% Compare false alarm rates
fprintf('\nFalse Alarm Rate Comparison:\n');
fprintf('Scenario 1: %.2f%%\n', false_alarm_rate_with_h);
fprintf('Scenario 2: %.2f%%\n', false_alarm_rate_without_h);
fprintf('Scenario 3: %.2f%%\n', false_alarm_rate_no_mtd);

% Average estimation error during attack
avg_error_during_attack = mean(error_norms(attack_period));
fprintf('\nAverage Estimation Error During Attack: %.4f\n', avg_error_during_attack);