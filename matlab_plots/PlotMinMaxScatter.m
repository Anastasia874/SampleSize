function PlotMinMaxScatter(dkl)

figure; hold on; plot(median(dkl,2)', 'k-', 'linewidth', 2)
%plot(max(dkl,[], 2)', 'k--', 'linewidth', 2)
%plot(min(dkl,[], 2)', 'k--', 'linewidth', 2)
plot(dkl, 'k.', 'markersize', 0.5)
axis tight
xlabel('Number of hist bars, $N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\chi^2$ divergence (disc)', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 20, 'FontName', 'Times')

end