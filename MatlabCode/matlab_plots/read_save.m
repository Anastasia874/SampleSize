function read_save 
%path(path,'C:\Users\Anastasia\Documents\Strijov\mlalgorithms\Group874\Motrenko2014KL\code');
dirname = fullfile('D:', 'Strijov', 'Multiscale\code\fig\frc_analysis\EnergyWeather\');
matfiles = dir(fullfile(dirname, 'qq*target*.eps'));

dirname = fullfile('C:\Users\motrenko\Documents\mlalgorithms\Group874\Motrenko2017ECoG\fig\scalogram_pictures\');
matfiles = dir(fullfile(dirname, 'scalo*.fig'));

path2pics = dirname;
save_path = dirname;
%save_path = 'C:/Users/Anastasia/Documents/Strijov/mlalgprithms/MLEducation/pics_png/';

for i = 1:length(matfiles)
   h = openfig([path2pics, matfiles(i).name]);
   saveas(h, [save_path, matfiles(i).name(1:end-4)], 'png');
   close(h);
   %[result,msg] = eps2xxx([path2pics, matfiles(i).name],{'png'}, 'C:\Program Files\gs\gs9.18\bin\gswin64c.exe');
end

end