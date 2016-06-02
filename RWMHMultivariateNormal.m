% Created by Seçil ÞEN
% Random Walk Metropolis Hastings for sampling from bivariate normal
% distribution

sigma = 0.55; % Sigma (For random walk behavior)
nSamples = 1000;
dimension = 16; 
x = zeros(dimension,nSamples);
x0 = zeros(1,dimension); x0(dimension) = 6; % Starting location for random walk.
x(:,1) = x0;
t = 2;

% Mean and variance for bivariate gaussian distribution
mu = zeros(1,dimension);
cov = diag(0.85:0.01:1);

% Generation of target distribution values
f_target = mvnrnd(mu, cov, nSamples)';

while(t < nSamples)
    y = randn(1,dimension,1)'*sigma + x(:,t-1); % Proposal generation (random walk)
    % Acceptance Rate calculation
    alpha = min(1,mvnpdf(transpose(y), mu, cov)/mvnpdf(x(:,t-1)', mu, cov));
    
    % Metropolis - Hastings accept/reject condition.
    u = rand(1);
    if(u < alpha)
        x(:,t) = y;
    else
        x(:,t) = x(:,t-1);
    end
    t = t + 1;
end

% Dimension reduction using PCA (from drtoolbox, see https://lvdmaaten.github.io/drtoolbox/#download)
[mappedX,mapping] = compute_mapping(x','PCA',2);
[mappedGauss, mappingGauss] = compute_mapping(f_target','PCA',2); % To display more points on figure. 

figure
scatter(mappedGauss(:,1),mappedGauss(:,2),'k.'); hold on;
plot(mappedX(1:50,1),mappedX(1:50,2),'ro-','Linewidth',2);
xlim([-6 6]); ylim([-6 6]);
legend({'Samples','1st 50 States'},'Location','Northwest')
title('Random Walk Metropolis Hastings')
