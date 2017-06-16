function plotOptimization(w, lhd, nIters, y, X, sigm)

% plots model likelihood and generalization error by iterations of
% optimization algorithm

[m, n] = size(X);
XX = [ones(m, 1), X];
%nIters = length(lhd);
[wd, wg] = wmu2wdg(w(1:nIters,:), sigm);
wd_cos = w(nIters:end, 1:n+1);
mu1_cos = w(nIters:end, n+2:2*n+1);
mu0_cos = w(nIters:end, 2*n+2:3*n+1);
wg_cos = [(mu1_cos-mu0_cos)/sigm, sum(mu1_cos.*mu1_cos - mu0_cos.*mu0_cos, 2)/sigm];


ge = zeros(nIters, 1);
ge_cos = zeros(length(w)-nIters,1);
for i = 1:nIters
   yy = XX*wd(i,:)' > 0; 
   ge(i) = mean(y ~= yy);
end
for i = 1:length(w)-nIters
   yy = XX*wd_cos(i,:)' > 0; 
   ge_cos(i) = mean(y ~= yy);
end
%{
plotWPairs(wd, wg);
plotWPairs(wd_cos, wg_cos);
plotGEiters(ge, lhd(1:nIters));
plotGEiters(ge_cos, lhd(nIters+1:end));
%}
nIters = max([nIters, length(w)]);
ge = [ge; ge(end)*ones(nIters-length(ge), 1)];
ge_cos = [ge_cos; ge_cos(end)*ones(nIters-length(ge_cos), 1)];

figure; hold on;
plot(1:nIters, ge', 'k-', 'linewidth', 2);
plot(1:nIters, ge_cos', 'k--', 'linewidth', 2)
hold off;
legend('Quadratic', 'Cosine');
axis tight
xlabel('Iterarions', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Error', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 20, 'FontName', 'Times')


end
