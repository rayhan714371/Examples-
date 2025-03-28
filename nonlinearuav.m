%% 6-DOF Nonlinear UAV Model with Moving Target Defense (MTD) for Attack Detection
% This implementation builds upon the paper's Moving Target Defense (MTD) approach
% but excludes the nonlinear transformation component.

%% Parameters and Initialization
dt = 0.01;                % Sampling time
T = 50;                   % Simulation time
N = T/dt;                 % Number of time steps
g = 9.81;                 % Gravity constant (m/s²)

% UAV Physical Parameters
m = 1.5;                  % Mass (kg)
Ixx = 0.0213;             % Moment of inertia around x-axis (kg·m²)
Iyy = 0.0213;             % Moment of inertia around y-axis (kg·m²)
Izz = 0.0373;             % Moment of inertia around z-axis (kg·m²)

% 6-DOF State Variables (12 states total)
% x = [x, y, z, φ, θ, ψ, u, v, w, p, q, r]'
% Where:
% [x, y, z]     - Position in inertial frame
% [φ, θ, ψ]     - Euler angles (roll, pitch, yaw)
% [u, v, w]     - Body-frame velocities
% [p, q, r]     - Angular rates

% Initialize state array
x = zeros(12, N+1);

% Set initial state
x(:,1) = [0; 0; -10; 0; 0; 0; 0; 0; 0; 0; 0; 0];  % Starting at 10m altitude

% Control input (4 inputs: thrust, roll torque, pitch torque, yaw torque)
u = zeros(4, N);

% LQR Controller Design
% Linearization point (hover): x_eq = [0 0 -h 0 0 0 0 0 0 0 0 0]', u_eq = [m*g 0 0 0]'
x_eq = [0; 0; -10; 0; 0; 0; 0; 0; 0; 0; 0; 0];
u_eq = [m*g; 0; 0; 0];

% Linearized system matrices (derived analytically for hover)
A = zeros(12, 12);
% Position derivatives depend on velocities through rotation matrix
A(1,7) = 1; A(2,8) = 1; A(3,9) = 1;  % At hover, this simplifies
% Euler angle derivatives depend on angular velocities through kinematic equations
A(4,10) = 1; A(5,11) = 1; A(6,12) = 1;  % At hover, this simplifies
% Acceleration terms (simplified for hover)
A(9,5) = -g;  % w-dot affected by pitch angle due to gravity

B = zeros(12, 4);
% Control effects on accelerations
B(7,1) = 1/m;      % u-dot affected by thrust 
B(10,2) = 1/Ixx;   % p-dot affected by roll torque
B(11,3) = 1/Iyy;   % q-dot affected by pitch torque
B(12,4) = 1/Izz;   % r-dot affected by yaw torque

% Output matrix - assume we can measure all states
C = eye(12);

% Measurement noise bounds
epsilon = 0.01;     % Bound on measurement noise

% LQR Controller Design
Q = diag([10 10 10 10 10 10 1 1 1 1 1 1]);  % State penalty
R = diag([0.1 1 1 1]);                      % Control penalty
[K_lqr, ~, ~] = lqr(A, B, Q, R);

% Construct full state observer
Q_obs = 10*eye(12);       % Process noise covariance
R_obs = 0.1*eye(12);      % Measurement noise covariance
L = lqr(A', C', Q_obs, R_obs)';  % Observer gain using dual LQR problem

% Moving Target Defense parameters
vartheta_min = 1.5;
vartheta_max = 2.0;

%% Attack Parameters
attack_start = 20/dt;     % Attack starts at t = 20s
attack_end = 30/dt;       % Attack ends at t = 30s
attack_magnitude = 0.1;   % Attack magnitude

% Detection threshold
r_threshold = 0.5;        % Detection threshold based on expected residuals

%% Initialize arrays for simulation
x_hat = zeros(12, N+1);   % State estimates
y = zeros(12, N+1);       % Measurements
yM = zeros(12, N+1);      % Moving target transformed measurements
y_bar = zeros(12, N+1);   % Restored measurements
r_a = zeros(12, N+1);     % Residuals
e = zeros(12, N+1);       % Estimation errors
attack_flag = zeros(1, N+1); % Attack detection flag
Gamma_k_values = zeros(12, N+1); % Store Gamma_k diagonal values
attacks = zeros(12, N+1); % Store attacks applied
h_nonlinear = zeros(12, N+1); % Nonlinear remnants after linearization

% Set initial state estimate 
x_hat(:,1) = x(:,1) + 0.1*randn(12,1); % Initial estimate with some uncertainty
e(:,1) = x(:,1) - x_hat(:,1);          % Initial estimation error

%% Main Simulation Loop
for k = 1:N
    % Current state
    x_k = x(:,k);
    
    % Desired position trajectory (circular path at constant altitude)
    t = (k-1)*dt;
    radius = 5;
    if t < 5  % Allow time to stabilize
        x_des = [0; 0; -10; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    else
        angle = 0.2*(t-5);  % Circle with 5s period
        x_des = [radius*cos(angle); radius*sin(angle); -10; 0; 0; angle; 0; 0; 0; 0; 0; 0];
    end
    
    % LQR Control input
    u(:,k) = u_eq - K_lqr * (x_k - x_des);
    
    % Generate nonlinear effects (treated as UBB noise)
    % In reality, this would be from linearization error and unmodeled dynamics
    h_nonlinear(:,k) = 0.05*randn(12,1) .* (1 + 0.1*sin(x_k));
    
    % True nonlinear system simulation using Euler integration
    % In a complete implementation, this would use the full nonlinear equations
    % Here we use the linearized model plus nonlinear remnant term
    x_dot = A * (x_k - x_eq) + B * (u(:,k) - u_eq) + h_nonlinear(:,k);
    x(:,k+1) = x_k + dt * x_dot;
    
    % Measurement noise
    v_k = (2*rand(12,1)-1) * epsilon;
    
    % Measurement
    y(:,k) = C * x(:,k) + v_k;
    
    % Create the Gamma_k matrix for Moving Target Defense
    Gamma_k_values(:,k) = vartheta_min + (vartheta_max - vartheta_min) * rand(12,1);
    Gamma_k = diag(Gamma_k_values(:,k));
    
    % Apply Moving Target Defense (without nonlinear transformation)
    yM(:,k) = Gamma_k * y(:,k);
    
    % Apply FDI attack if in attack interval
    attacks(:,k) = zeros(12,1);
    if k >= attack_start && k <= attack_end
        % Attack model: targeted attack on position measurements
        attacks([1,2,3],k) = attack_magnitude * ones(3,1);
    end
    ya_k = yM(:,k) + attacks(:,k);
    
    % Restore original measurement (apply MTD inverse)
    y_bar(:,k) = Gamma_k \ ya_k;
    
    % State estimation using Luenberger observer
    x_hat_dot = A * (x_hat(:,k) - x_eq) + B * (u(:,k) - u_eq) + L * (y_bar(:,k) - C * x_hat(:,k));
    x_hat(:,k+1) = x_hat(:,k) + dt * x_hat_dot;
    
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

%% Plotting Results
time = (0:N)*dt;

% Figure 1: 3D Trajectory
figure;
plot3(x(1,:), x(2,:), -x(3,:), 'b-', 'LineWidth', 2);
hold on;
plot3(x(1,1), x(2,1), -x(3,1), 'go', 'MarkerSize', 10, 'LineWidth', 2);  % Initial point

% Mark attack period with different color
attack_indices = attack_start:attack_end;
plot3(x(1,attack_indices), x(2,attack_indices), -x(3,attack_indices), 'r-', 'LineWidth', 2.5);

% Plot estimated trajectory
plot3(x_hat(1,:), x_hat(2,:), -x_hat(3,:), 'c--', 'LineWidth', 1.5);

title('Figure 1: UAV 3D Trajectory');
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
legend('Actual Trajectory', 'Initial Position', 'Under Attack', 'Estimated Trajectory');
grid on;

% Figure 2: Position vs Time
figure;
subplot(3,1,1);
plot(time, x(1,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(time, x_hat(1,:), 'c--', 'LineWidth', 1.5);
title('Figure 2: Position vs Time');
ylabel('X Position (m)');
grid on;

% Add shaded region for attack period
attack_region_x = [time(attack_start), time(attack_end), time(attack_end), time(attack_start)];
attack_region_y = [min(x(1,:))-1, min(x(1,:))-1, max(x(1,:))+1, max(x(1,:))+1];
patch(attack_region_x, attack_region_y, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

subplot(3,1,2);
plot(time, x(2,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(time, x_hat(2,:), 'c--', 'LineWidth', 1.5);
ylabel('Y Position (m)');
grid on;

% Add shaded region for attack period
attack_region_y = [min(x(2,:))-1, min(x(2,:))-1, max(x(2,:))+1, max(x(2,:))+1];
patch(attack_region_x, attack_region_y, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

subplot(3,1,3);
plot(time, -x(3,:), 'b-', 'LineWidth', 1.5);
hold on;
plot(time, -x_hat(3,:), 'c--', 'LineWidth', 1.5);
ylabel('Z Position (m)');
xlabel('Time (s)');
legend('Actual', 'Estimated');
grid on;

% Add shaded region for attack period
attack_region_y = [min(-x(3,:))-1, min(-x(3,:))-1, max(-x(3,:))+1, max(-x(3,:))+1];
patch(attack_region_x, attack_region_y, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
text((time(attack_start) + time(attack_end))/2, min(-x(3,:))-0.5, 'Attack Period', 'HorizontalAlignment', 'center');

% Figure 3: Orientation (Euler angles)
figure;
subplot(3,1,1);
plot(time, rad2deg(x(4,:)), 'b-', 'LineWidth', 1.5);
hold on;
plot(time, rad2deg(x_hat(4,:)), 'c--', 'LineWidth', 1.5);
title('Figure 3: Orientation (Euler Angles)');
ylabel('Roll (deg)');
grid on;

subplot(3,1,2);
plot(time, rad2deg(x(5,:)), 'b-', 'LineWidth', 1.5);
hold on;
plot(time, rad2deg(x_hat(5,:)), 'c--', 'LineWidth', 1.5);
ylabel('Pitch (deg)');
grid on;

subplot(3,1,3);
plot(time, rad2deg(x(6,:)), 'b-', 'LineWidth', 1.5);
hold on;
plot(time, rad2deg(x_hat(6,:)), 'c--', 'LineWidth', 1.5);
ylabel('Yaw (deg)');
xlabel('Time (s)');
legend('Actual', 'Estimated');
grid on;

% Figure 4: Estimation Error (Position)
figure;
subplot(3,1,1);
plot(time, e(1,:), 'b-', 'LineWidth', 1.5);
title('Figure 4: Position Estimation Error');
ylabel('X Error (m)');
grid on;

subplot(3,1,2);
plot(time, e(2,:), 'b-', 'LineWidth', 1.5);
ylabel('Y Error (m)');
grid on;

subplot(3,1,3);
plot(time, e(3,:), 'b-', 'LineWidth', 1.5);
ylabel('Z Error (m)');
xlabel('Time (s)');
grid on;

% Add shaded regions for attack period across all subplots
for i = 1:3
    subplot(3,1,i);
    attack_region_x = [time(attack_start), time(attack_end), time(attack_end), time(attack_start)];
    attack_region_y = [min(e(i,:))-0.5, min(e(i,:))-0.5, max(e(i,:))+0.5, max(e(i,:))+0.5];
    hold on;
    patch(attack_region_x, attack_region_y, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end

% Figure 5: Attack Detection Results
figure;
stem(time, attack_flag, 'filled', 'LineWidth', 1.5);
title('Figure 5: Attack Detection Results');
xlabel('Time (s)');
ylabel('Attack Flag');
ylim([-0.1, 1.1]);
yticks([0 1]);
yticklabels({'No Attack', 'Attack Detected'});
grid on;

% Add shaded region for attack period
hold on;
attack_region_x = [time(attack_start), time(attack_end), time(attack_end), time(attack_start)];
attack_region_y = [-0.1, -0.1, 1.1, 1.1];
patch(attack_region_x, attack_region_y, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
text((time(attack_start) + time(attack_end))/2, -0.05, 'Actual Attack Period', 'HorizontalAlignment', 'center');

% Figure 6: Moving Target Defense Parameter Values
figure;
plot(time(1:N), Gamma_k_values(:,1:N)', 'LineWidth', 1);
title('Figure 6: Moving Target Defense Parameters (Γ_k values)');
xlabel('Time (s)');
ylabel('Parameter Value');
grid on;
ylim([vartheta_min-0.1, vartheta_max+0.1]);

% Figure 7: Residuals
figure;
plot(time(1:N), vecnorm(r_a(:,1:N)), 'b-', 'LineWidth', 1.5);
hold on;
plot(time(1:N), r_threshold*ones(size(time(1:N))), 'r--', 'LineWidth', 1.5);
title('Figure 7: Residual Norm for Attack Detection');
xlabel('Time (s)');
ylabel('||r_a||');
grid on;
legend('Residual Norm', 'Detection Threshold');

% Add shaded region for attack period
attack_region_x = [time(attack_start), time(attack_end), time(attack_end), time(attack_start)];
attack_region_y = [0, 0, max(vecnorm(r_a(:,1:N)))+0.5, max(vecnorm(r_a(:,1:N)))+0.5];
patch(attack_region_x, attack_region_y, 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');