% Created by Seçil ÞEN
% Random Walk Metropolis Hastings for sampling from bivariate normal
% distribution with mean mu = [0,0] and covariance cov = [1,0.8; 0.8,1]

sigma = 0.55; % Sigma (For random walk behavior)
nSamples = 1000;
x = zeros(2,nSamples);
x0 = [0;6]; % Starting location for random walk.
x(:,1) = x0;
t = 2;

% Mean and variance for bivariate gaussian distribution
mu = [0,0];
cov = [1,0.8;0.8,1];

% Generation of target distribution values
f_target = mvnrnd(mu, cov, nSamples)';

while(t < nSamples)
    y = randn(1,2,1)'*sigma + x(:,t-1); % Proposal generation (random walk)
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

figure
scatter(f_target(1,:),f_target(2,:),'k.'); hold on;
plot(x(1,1:50),x(2,1:50),'ro-','Linewidth',2);
xlim([-6 6]); ylim([-6 6]);
legend({'Samples','1st 50 States'},'Location','Northwest')
title('Random Walk Metropolis Hastings')
