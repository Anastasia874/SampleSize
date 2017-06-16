function myPlot2Y(ts1, ts2, args)

t = 1:length(ts1);
txt_xlabel = 'Time';
txt_ylabel1 = 'Time series 1';
txt_ylabel2 = 'Time series 2';
name = [];
i = 1;
while i < length(args)
   switch args{i} 
       case 'ts'
           ts = args{i+1};
           i = i+2;
       case 'x'
           t = args{i+1};
           i = i+2;
       case 'xlbl'
           txt_xlabel = args{i+1};
           i = i+2;   
       case 'ylbl1'
           txt_ylabel1 = args{i+1};
           i = i+2; 
       case 'ylbl2'
           txt_ylabel2 = args{i+1};
           i = i+2;     
       case 'saveas'
           name = args{i+1};
           i = i+2; 
       otherwise
           i = i+1;
   end
end

h = figure;
[AX, H1, H2] = plotyy(t, ts1, t, ts2);%, 'plot', 'semilogy');
set(H1,'LineStyle','-','color','blue', 'linewidth', 0.5);
set(H2,'LineStyle','--','color','black', 'linewidth', 1);
set(AX,{'ycolor'},{'k';'k'}) 
set(get(AX(1),'Ylabel'),'String', txt_ylabel1, 'Interpreter', 'Latex');
set(get(AX(2),'Ylabel'),'String', txt_ylabel2, 'Interpreter', 'Latex'); %'LogLikelihood');
set(AX,'FontSize',20, 'Fontname', 'Times')
axis(AX, 'tight');
set(get(AX(1),'Xlabel'),'String', txt_xlabel, 'Interpreter', 'Latex'); %'Iterations');
%set(AX(1), 'Xtick', [1:5:21]);
%set(AX(1), 'XtickLabel', [0:0.25:1]);
set(AX(2), 'Xtick', []);
if ~isempty(name)
    saveas(h, [name, '.fig']);
    close(h);
end
end