function plotGEiters(ge, lhd)
nIters = length(ge);
figure;
[AX, H1, H2] = plotyy(1:nIters, ge', 1:nIters, lhd');%, 'plot', 'semilogy');
set(H1,'LineStyle','-','color','black', 'linewidth', 2);
set(H2,'LineStyle','--','color','black', 'linewidth', 2);
set(AX,{'ycolor'},{'k';'k'}) 
set(get(AX(1),'Ylabel'),'String','Generalization error');
set(get(AX(2),'Ylabel'),'String', 'Test statistic, $t_m$', 'Interpreter', 'Latex'); %'LogLikelihood');
set(AX,'FontSize',20, 'Fontname', 'Times')
axis(AX, 'tight');
set(get(AX(1),'Xlabel'),'String', '$\lambda$', 'Interpreter', 'Latex'); %'Iterations');
set(AX(1), 'Xtick', [1:5:21]);
set(AX(1), 'XtickLabel', [0:0.25:1]);
set(AX(2), 'Xtick', []);
end