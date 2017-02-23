function plot_gaussian_ellipse(S,mu,n,style,num_points)
% plot_gaussian_ellipse(S,mu)
% plot_gaussian_ellipse(S,mu,n)
% plot_gaussian_ellipse(S,mu,n,style)
% plot_gaussian_ellipse(S,mu,n,style,num_points)

if nargin==2; n = exp(-1); end
if nargin<=3; style=''; end
if nargin<=4; num_points=1000; end

n=n*ones(size(mu,2),1);
theta = 0:2*pi/(num_points-1):2*pi;
u = [cos(theta);sin(theta)];
X = zeros(numel(theta),numel(n));
Y = zeros(numel(theta),numel(n));
for i=1:numel(n)
  Si = inv(S(:,:,i));
  r = sqrt((n(i)/2)./(...
    Si(1,1)*u(1,:).^2+...
    Si(2,2)*u(2,:).^2+...
    2*Si(1,2)*(u(1,:).*u(2,:))));
  X(:,i) = mu(1,i)+u(1,:)'.*r(:);
  Y(:,i) = mu(2,i)+u(2,:)'.*r(:);
end

plot(X,Y,style,'LineWidth',2);