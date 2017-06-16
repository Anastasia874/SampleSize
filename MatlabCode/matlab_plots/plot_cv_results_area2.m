function plot_cv_results_area2(mean_vals, std_vals, x_vals, ytext, legend_text, name)

                           
ALPHA = 0.2;                   
if nargin < 5
    legend_text = {'Train', 'Test'};
end
if nargin < 6
    name = [];
end
colormap summer;

n_ts = numel(mean_vals);
fh = {};
lh = zeros(1, n_ts);

                           
f = figure; hold on;
for i = 1:n_ts
h = fill([x_vals, fliplr(x_vals)], [mean_vals{i} - std_vals{i}, ...
      fliplr(mean_vals{i} + std_vals{i})], 'o');
h.FaceAlpha = ALPHA;
h.EdgeColor = 'none';
fh{end+1} = h.FaceColor;
end

axis tight;
xlabel('Number of PLS components', 'FontSize', 20, 'FontName', 'Times', ...
    'Interpreter','latex');
ylabel(ytext, 'FontSize', 20, 'FontName', 'Times', ...
    'Interpreter','latex');
set(gca, 'FontSize', 15, 'FontName', 'Times');

% [hl, icons,~,~] = legend(fh, legend_text, 'location', 'best', 'fontname', 'Times', 'fontsize', 15);
% pos = hl.Position;

specs = {'-', '--', ':', '-', '-.', '-', '-', '-', '-'};
markers = {'none', 'none', 'none', 'x', 'none', '+', 'o', '*', '^'};
% specs = {'-', '--', ':', '-x', '-.', '-+', '-o'};
for i = 1:n_ts
lh(i) = plot(x_vals, mean_vals{i}, 'linestyle', specs{i}, 'marker', markers{i},...
                               'color', fh{i}, 'linewidth', 2, 'markersize', 4);
% icons(i + n_ts).FaceAlpha = ALPHA;
% xp = icons(i + n_ts).Vertices([1, 3], 1);
% yp = icons(i + n_ts).Vertices(1:2, 2);
% l = annotation('line');
% l.X = xp * pos(2) + pos(1);
% l.Y = [1 1]*mean(yp)*pos(4) + pos(2);
% l.LineStyle = specs{i};
% l.Color = 'k';
% l.LineWidth = 2;
% l.Marker = markers{i};
% l.MarkerSize = 4;
% l.Units = 'pixels';
end
legend(lh, legend_text, 'location', 'best', 'fontname', 'Times', 'fontsize', 15);
hold off;



if ~isempty(name)
savefig(['../fig/results/', name, '.fig']);
saveas(f, ['../fig/results/', name, '.png'])
close(f);
end

end