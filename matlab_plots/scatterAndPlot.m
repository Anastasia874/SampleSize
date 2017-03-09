function scatterAndPlot(ts, args)

t = 1:length(ts);
txt_xlabel = 'Time';
txt_ylabel = 'Time series';
specs = {'k-', 'k.'}; %'k--', 'k:', 'k-.'};
txt_legend = [];
av_func = 'median';
name = [];
i = 1;
while i < length(args)
   switch args{i} 
       case 'avfunc'
           av_func = args{i+1};
           i = i+2;
       case 'x'
           t = args{i+1};
           i = i+2;
       case 'xlbl'
           txt_xlabel = args{i+1};
           i = i+2;   
       case 'ylbl'
           txt_ylabel = args{i+1};
           i = i+2; 
       case 'legend'
           txt_legend = args{i+1};
           i = i+2;    
       case 'specs'
           specs = args{i+1};
           i = i+2;        
       case 'saveas'
           name = args{i+1};
           i = i+2; 
       otherwise
           i = i+1;
   end
end

av_ts = feval(av_func, ts);

h = figure;
hold on;
plot(t, av_ts, specs{1}, 'linewidth', 2);
plot(t, ts, specs{2}, 'linewidth', 2);
hold off;
xlabel(txt_xlabel, 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel(txt_ylabel, 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 20, 'FontName', 'Times')
axis tight;
if ~isempty(txt_legend)
    lh = legend(txt_legend);
    set(lh, 'Interpreter', 'Latex');
end
if ~isempty(name)
    saveas(h, [name, '.fig']);
    close(h);
end
end