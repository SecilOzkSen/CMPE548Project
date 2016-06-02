
delta = 0.85; % Step size
nSamples = 1000; % Number of samples
L = 30; % Leapfrog size
dimension = 16; % Dimension of multivariate gaussian distribution

% covariance matrix of multivariate gaussian distribution (mean = [0,....0])
cov = diag(0.85:0.01:1);

% States and initial state
x = zeros(dimension,nSamples);
% Start state
x0 = zeros(1,dimension); x0(dimension) = 6;
x(:,1) = x0;

% Potential Energy Function
U = inline('transpose(x)*inv(cov)*x','x','cov');

% Gradient of potential energy
dU = inline('transpose(x)*inv(cov)','x','cov');

% Kinetic Energy Function
K = inline('sum((transpose(p)*p))/2','p');

% Hamiltonian
H = zeros(1,nSamples);
% Gaussian
gauss = mvnrnd(zeros(1,dimension),cov,nSamples);

t = 1;
while(t < nSamples)
    t = t + 1;
    % Sample random momentum
    p0 = randn(dimension,1);
    %%%%%%  SIMULATE HAMILTONIAN DYNAMICS %%%%%%
    %%% LEAPFROG - START %%%
    % Step 1 - First 1/2 step of momentum
    pStar = p0 - delta/2*dU(x(:,t-1),cov)';
    
    % Step 2a - First full step for position/sample
    xStar = x(:,t-1) + delta*pStar;
    
    % Step 2b - Full step for go from p0 to pStar
    for i = 1:L-1
        pStar = pStar - delta*dU(xStar,cov)'; % Momentum
        xStar = xStar + delta*pStar; %position
    end
    % Step 3 - Last half step
    pStar = pStar - delta/2*dU(xStar,cov)';
    
    %%% LEAPFROG - END %%%
    
    % Evaluate Kinetic and Potential Energies at Start and End of
    % Trajectory.
    U0 = U(x(:,t-1),cov);
    UStar = U(xStar,cov);
    
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

% Dimension reduction using PCA (from drtoolbox, see https://lvdmaaten.github.io/drtoolbox/#download)
[mappedX,mapping] = compute_mapping(x','PCA',2);
[mappedGauss, mappingGauss] = compute_mapping(gauss,'PCA',2); % To display more points on figure. 
disp(mappedX);

% Display
figure;
scatter(mappedGauss(:,1),mappedGauss(:,2),'k.'); hold on; % to see more points.
plot(mappedX(1:50,1),mappedX(1:50,2),'ro-','Linewidth',2);
xlim([-4 4]); ylim([-4 4]);
legend({'Samples','1st 50 States'},'Location','Northwest')
title('Hamiltonian Monte Carlo (Dimension = 16)')
