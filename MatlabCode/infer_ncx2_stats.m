function [h, p, C, N, delta, chi2fit] = infer_ncx2_stats(x, N, sgnf)
% x ~ C*chi^2_N(Delta)
% infer constant C and noncetr. parameter Delta of fr. degrees from mean and variance:
N_BARS = 15;
C = var(x)/2/mean(x);
[vd, conf_int] = mle(x/C,'pdf',@(x,v,d)ncx2pdf(x,v,d),'start',[N/2,1]);
N = vd(1);
delta = vd(2);

[nX, cX] = hist(x/c, N_BARS);
x2pdf = ncx2pdf(cX, N);
chi2fit = mean((nX/length(x) - x2pdf).^2);

figure;
bar(cX, nX/length(x));
hold on;
plot(x, ncx2/sum(ncx2), 'g-', 'linewidth', 2);
%plot(x, rnd/sum(rnd), 'g--', 'linewidth', 2);
hold off;
h = legend('$\hat{P}(D_{KL})$', 'n.c.$\chi^2$');
set(h, 'Interpreter', 'latex')
axis tight;
xlabel('$x$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$P(D_{KL}| H_0)$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 20, 'FontName', 'Times')
set(gca, 'XTick', [])

end