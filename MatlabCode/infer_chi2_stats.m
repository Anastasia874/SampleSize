function [hp, p, C, N, chi2fit] = infer_chi2_stats(x, significance, name)
% x ~ C*chi^2_N
% infer constant C and nunmber N of fr. degrees from mean and variance:
N_BARS = 10;
C = var(x)/2/mean(x);
N = mean(x)/C;
%{
[NC, conf_int] = mle(x,'pdf',@(x,v,c)chi2pdf(x/c,v/c),'start', [1, 1]);
N = NC(1);
C = NC(2);
%}
% compare empirical pdf(x/C) to pdf(chi2_N)
[nX, cX] = hist(x/C, N_BARS);
x2pdf = chi2pdf(cX, N);
chi2fit = mean((nX/length(x) - x2pdf).^2./x2pdf);

%{
% infer parameters via gamma approximation
C = var(x)/2/mean(x);
[a_gamma, b_gamma] = fitdist(x(:)/2, 'Gamma');
%}

p = chi2cdf(chi2fit, N_BARS);
hp = chi2fit > chi2inv(1 - significance, N_BARS);


%{
fig = figure;
bar(cX, nX/length(x));
hold on;
plot(cX, x2pdf/sum(x2pdf), 'g-', 'linewidth', 2);
%plot(x, rnd/sum(rnd), 'g--', 'linewidth', 2);
hold off;
h = legend('$\hat{P}(D_{KL})$', '$\chi^2$');
set(h, 'Interpreter', 'latex')
axis tight;
xlabel('$x$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$P(D_{KL}| H_1)$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 20, 'FontName', 'Times')
set(gca, 'XTick', [])
saveas(fig, [name, '.fig']);
close(fig);
%}
end