X0=[1;0;0]; %Initial conditions

A=[0 1 0;
   0 -1 50;
   0 -20 100]; %System dynamics

B=[0;0;1];

C=[1 0 0; 0 1 0; 0 0 1];  

D=[0; 0; 0];

%Controller
Q=[1 0 0;  %Penalize x1 error
   0 1 0;  %Penalize x2 error
   0 0 1]; %Penalize x3 error

R=0.01;   %Penalize actuator effort

K=lqr(A,B,Q,R);

%Closed loop system
sys=ss((A-B*K),B,C,D);

%Run response to initial condition
[Y,T,X]=initial(sys, X0);

%Create figures
if ~exist('h1', 'var')
    h1=figure('Position',[100 670 670 238]);
    title('Response to initial condition', 'FontSize', 14);
    h1.ToolBar='none';
end
if ~exist('h2', 'var')
    h2=figure('Position',[100 390 670 234]);
    title('Actuator effort', 'FontSize', 14);
    h2.ToolBar='none';
end
if ~exist('h3', 'var')
    h3=figure('Position',[100 80 670 236]);
    title('Pole / Zero Map', 'FontSize', 14);
    h3.ToolBar='none';
end

%Plot response 
set(0, 'currentfigure', h1);
hold all
p1 = plot(T,Y(:, 1), 'LineWidth', 4);

%Plot actuator effort
set(0, 'currentfigure', h2);
hold all
p2= plot(T, -K*X', 'LineWidth', 4);

%Plot poles and zeros
set(0, 'currentfigure', h3);
hold all
pzmap(sys);