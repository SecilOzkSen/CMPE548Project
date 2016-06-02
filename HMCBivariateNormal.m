% Created by Seçil ÞEN
% Hamiltonian Monte Carlo for Sampling from Bivariate Normal Distribution
% with mean mu = [0,0] and covariance cov = [1,0.8; 0.8,1]

delta = 0.3; % Step Size
nSamples = 1000; % Number of Samples
L = 20; % Leapfrog Size

% Potential Energy Function
U = inline('transpose(x)*inv([1,0.8;0.8,1])*x', 'x'); % U = x^T*Sigma^-1*x

% Gradient of Potential Energy
dU = inline('transpose(x)*inv([1,0.8;0.8,1])', 'x');

% Kinetic Energy Function
K = inline('sum((transpose(p)*p))/2', 'p');

% States and initial state
x = zeros(2,nSamples);
x0 = [0;6]; % initial state (start state)
x(:,1) = x0;
% Multivariate gaussian dist.
gauss = mvnrnd([0,0],[1,0.8;0.8,1],nSamples);
% Hamiltonian
H = zeros(1,nSamples);
t = 1;
while(t < nSamples)
    t = t + 1;
    % Sample random momentum
    p0 = randn(2,1);
    
    %%%%%%  SIMULATE HAMILTONIAN DYNAMICS %%%%%%
    %%% LEAPFROG - START %%%
    % Step 1 - First 1/2 step of momentum
    pStar = p0 - delta/2*dU(x(:,t-1))';
    
    % Step 2a - First full step for position/sample
    xStar = x(:,t-1) + delta*pStar;
    
    % Step 2b - Full step for go from p0 to pStar
    for i = 1:L-1
        pStar = pStar - delta*dU(xStar)'; % Momentum
        xStar = xStar + delta*pStar; %position
    end
    % Step 3 - Last half step
    pStar = pStar - delta/2*dU(xStar)';
    
    %%% LEAPFROG - END %%%
    
    % Evaluate Kinetic and Potential Energies at Start and End of
    % Trajectory.
    U0 = U(x(:,t-1));
    UStar = U(xStar);
    
    K0 = K(p0);
    KStar = K(pStar);
    
    % Accept or reject
    alpha = min(1, exp((U0 + K0) - (UStar + KStar)));
    u = rand(1);
    
    if(u < alpha)
        x(:,t) = xStar;
        % Hamiltonian value
        H(t-1) = UStar + KStar;
    else
        x(:,t) = x(:,t-1);
        % Hamiltonian value
        H(t-1) = U0 + K0; 
    end
    
end

% Display
scatter(x(1,:),x(2,:),'k.'); hold on;
plot(x(1,1:30),x(2,1:30),'ro-','Linewidth',2);
xlim([-6 6]); ylim([-6 6]);
legend({'Samples','1st 50 States'},'Location','Northwest')
title('Hamiltonian Monte Carlo')


%figure(1);
%scatter(P(1,:),P(2,:),'k.'); hold on;
%plot(P(1,1:50),P(2,1:50),'ro-','Linewidth',2);
%xlim([-6 6]); ylim([-6 6]);
%legend({'Samples','1st 50 States'},'Location','Northwest')
%title('Hamiltonian Monte Carlo')
