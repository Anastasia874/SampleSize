function plotCls(X, y, args)

txt_xlabel = '$x_1$';
txt_ylabel = '$x_2$';
colors = {'r', 'b', 'k', 'g'};
y_pred = [];
txt_legend = [];
name = [];
i = 1;
while i < length(args)
   switch args{i} 
       case 'res'
           y_pred = args{i+1};
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
           colors = args{i+1};
           i = i+2;        
       case 'saveas'
           name = args{i+1};
           i = i+2; 
       otherwise
           i = i+1;
   end
end

figure;
hold on;
i = 0;
for cls = unique(y)'
    i = i+1;
    plot(X(y == cls, 1), X(y == cls, 2), [colors{i},'.'], ...
                                    'markersize', 10);
end
if ~isempty(y_pred)
    i = 0;
   for cls = unique(y_pred)'
    i = i+1;
    plot(X(y_pred == cls, 1), X(y_pred == cls, 2), [colors{i},'o'], ...
                                    'markersize', 7, 'linewidth', 1.5);
    end 
end
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