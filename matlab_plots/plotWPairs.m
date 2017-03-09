function plotWPairs(wd, wg)
% For 2D parametrs (one-dimensional feature space)
% plots a sequaence of generative-discriminative pairs

nIters = length(wd);
figure; hold on;
plot(wd(:, 1), wd(:, 2), 'r.', 'linewidth', 2, 'markersize', 20);
plot(wg(:, 1), wg(:, 2), 'b.', 'linewidth', 2, 'markersize', 20);
h = legend('$\mathbf{w}_D$', '$\mathbf{w}_G$');
set(h, 'Interpreter', 'latex');
%plot(0, 0, 'ko','linewidth', 2, 'markersize', 20);
plot(wd(1, 1), wd(1, 2), 'ro', 'linewidth', 2, 'markersize', 20);
plot(wg(1, 1), wg(1, 2), 'bo', 'linewidth', 2, 'markersize', 20);
plot(wd(end, 1), wd(end, 2), 'rx', 'linewidth', 2, 'markersize', 20);
plot(wg(end, 1), wg(end, 2), 'bx', 'linewidth', 2, 'markersize', 20);
for i = 1:nIters
    plot([wd(i, 1), wg(i, 1)], [wd(i, 2), wg(i, 2)], 'k-', 'linewidth', 2);
end 
hold off;
axis tight
xlabel('$\beta$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$c$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 20, 'FontName', 'Times')
end 