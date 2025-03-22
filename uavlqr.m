%% UAV 6-DOF LQR Control Example
% This example shows how to implement an LQR controller
% for a linearized 6-DOF UAV model

% Define the linearized UAV model state space matrices
% State vector: [x, y, z, u, v, w, phi, theta, psi, p, q, r]
% x,y,z: position, u,v,w: velocity, phi,theta,psi: attitude, p,q,r: angular rates
% Control inputs: [throttle, aileron, elevator, rudder]

% System matrices (these would come from your linearization)
% Example values only - replace with your actual linearized model
A = [0 0 0 1 0 0 0 0 0 0 0 0;
     0 0 0 0 1 0 0 0 0 0 0 0;
     0 0 0 0 0 1 0 0 0 0 0 0;
     0 0 0 -0.1 0 0 0 -9.81 0 0 0 0;
     0 0 0 0 0 -0.1 9.81 0 0 0 0 0;
     0 0 0 0 0 0 0 -0.1 0 0 0 0;
     0 0 0 0 0 0 0 0 0 1 0 0;
     0 0 0 0 0 0 0 0 0 0 1 0;
     0 0 0 0 0 0 0 0 0 0 0 1;
     0 0 0 0 0 0 0 0 0 -1 0 0;
     0 0 0 0 0 0 0 0 0 0 -1 0;
     0 0 0 0 0 0 0 0 0 0 0 -1;];
% Position dynamics: A(1,4) = 1; A(2,5) = 1; A(3,6) = 1;
% Velocity dynamics (simplified): A(4,4) = -0.1; A(5,5) = -0.1; A(6,6) = -0.1;
%A(4,8) = -9.81; A(5,7) = 9.81; % Gravity effects
% Attitude kinematics:A(7,10) = 1; A(8,11) = 1; A(9,12) = 1;
% Angular velocity dynamics (simplified):A(10,10) = -1; A(11,11) = -1; A(12,12) = -1;

% Control input matrix
B = [0 0 0 0;
     0 0 0 0;
     0 0 0 0;
     0 0 0 0;
     0 0 0 0;
     1 0 0 0;
     0 0 0 0;
     0 0 0 0;
     0 0 0 0;
     0 5 0 0;
     0 0 5 0;
     0 0 0 5;];
% Thrust affects z velocity: B(6,1) = 1; 
% Control surfaces affect angular accelerations
%B(10,2) = 5; % Aileron → roll rate
%B(11,3) = 5; % Elevator → pitch rate 
%B(12,4) = 5; % Rudder → yaw rate

% Output matrix (if we measure all states)
C = eye(12);

% Direct feedthrough matrix
D = zeros(12,4);

% Initial conditions (UAV at position origin, slight attitude disturbance)
X0 = [0; 0; 0; 0; 0; 0; 0.1; 0.1; 0; 0; 0; 0];

% Define desired state (e.g., hover at position [10,10,-5])
Xd = [10; 10; -5; 0; 0; 0; 0; 0; 0; 0; 0; 0];

%% LQR Controller Design
% State penalty matrix Q (diagonal matrix with weights)
Q =[10 0 0 0 0 0 0 0 0 0 0 0;
    0 10 0 0 0 0 0 0 0 0 0 0;
    0 0 10 0 0 0 0 0 0 0 0 0;     % Position errors (x,y,z)
    0 0 0 1 0 0 0 0 0 0 0 0;
    0 0 0 0 1 0 0 0 0 0 0 0; 
    0 0 0 0 0 1 0 0 0 0 0 0;        % Velocity errors (u,v,w)
    0 0 0 0 0 0 5 0 0 0 0 0;
    0 0 0 0 0 0 0 5 0 0 0 0;
    0 0 0 0 0 0 0 0 1 0 0 0;        % Attitude errors (phi,theta,psi)
    0 0 0 0 0 0 0 0 0 1 0 0;
    0 0 0 0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 0 0 0 0.1];       % Angular rate errors (p,q,r)


% Control penalty matrix R
R =[0.1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1];  % Throttle, Aileron, Elevator, Rudder

% Compute the LQR gain matrix K
K = lqr(A, B, Q, R);

% Create the closed loop system
sys_cl = ss(A-B*K, B, C, D);

%% Simulation
% Simulation time and step
t_final = 10;  % seconds
dt = 0.01;     % time step
t = 0:dt:t_final;

% Initialize arrays
n = length(t);
X = zeros(12, n);
U = zeros(4, n);
X(:,1) = X0;

% Simple simulation loop (forward Euler integration)
for i = 1:n-1
    % Current state error
    error = X(:,i) - Xd;
    
    % Compute control input using LQR
    U(:,i) = -K * error;
    
    % Saturate control inputs to realistic bounds
    U(1,i) = max(0, min(1, U(1,i)));        % Throttle: 0 to 1
    U(2:4,i) = max(-0.5, min(0.5, U(2:4,i))); % Control surfaces: -0.5 to 0.5
    
    % Update state (simplified integration)
    X_dot = A*X(:,i) + B*U(:,i);
    X(:,i+1) = X(:,i) + X_dot*dt;
end

%% Visualize Results
% Create figures
figure('Position', [100 100 800 600]);
subplot(3,1,1)
plot(t, X(1:3,:)', 'LineWidth', 2)
grid on
legend('x', 'y', 'z')
title('Position')
ylabel('Position (m)')

subplot(3,1,2)
plot(t, X(7:9,:)'*180/pi, 'LineWidth', 2)
grid on
legend('\phi', '\theta', '\psi')
title('Attitude')
ylabel('Angle (deg)')

subplot(3,1,3)
plot(t, U', 'LineWidth', 2)
grid on
legend('Throttle', 'Aileron', 'Elevator', 'Rudder')
title('Control Inputs')
xlabel('Time (s)')
ylabel('Control input')

% 3D trajectory visualization
figure('Position', [900 100 600 600]);
plot3(X(1,:), X(2,:), -X(3,:), 'r-', 'LineWidth', 2)
hold on
plot3(Xd(1), Xd(2), -Xd(3), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b')
plot3(X0(1), X0(2), -X0(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g')
grid on
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
title('UAV Trajectory')
legend('Trajectory', 'Target', 'Start')
view(30, 30)

% Add UAV orientation at several points
num_markers = 5;
indices = round(linspace(1, n, num_markers));
for idx = indices
    % Plot a simple marker for UAV orientation
    plotUAVMarker(X(1,idx), X(2,idx), -X(3,idx), X(7,idx), X(8,idx), X(9,idx), 0.5);
end

%% Helper function to visualize UAV orientation
function plotUAVMarker(x, y, z, phi, theta, psi, scale)
    % Create a simple marker for UAV orientation
    % Creating rotation matrix from Euler angles
    R = eul2rotm([psi, theta, phi], 'ZYX');
    
    % Create UAV body axes
    body_x = scale * [1; 0; 0];
    body_y = scale * [0; 1; 0]; 
    body_z = scale * [0; 0; 1];
    
    % Rotate body axes
    body_x_rotated = R * body_x;
    body_y_rotated = R * body_y;
    body_z_rotated = R * body_z;
    
    % Plot the rotated axes
    quiver3(x, y, z, body_x_rotated(1), body_x_rotated(2), body_x_rotated(3), 'r', 'LineWidth', 2);
    quiver3(x, y, z, body_y_rotated(1), body_y_rotated(2), body_y_rotated(3), 'g', 'LineWidth', 2);
    quiver3(x, y, z, body_z_rotated(1), body_z_rotated(2), body_z_rotated(3), 'b', 'LineWidth', 2);
end