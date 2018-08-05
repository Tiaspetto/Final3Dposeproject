PathRoot = 'D:/dissertation/data/human3.6/S1/MyPoseFeatures/D2_Positions/';
SaveRoot = 'D:/dissertation/data/human3.6/S1/MyPoseFeatures/processed_2D/';
savetype = '.mat'
list=dir(fullfile(PathRoot));

fileNum=size(list,1)-2

for k=3:fileNum
    file = list(k).name  % 这就是文件名，如果有子文件夹，则也包含在里面。
    if strfind(file,'.cdf')
        file_name = erase(file,'.cdf')
        file_name = [file_name, savetype]
        data = cdfread([PathRoot,file])
        data = data{1,1}
        save([SaveRoot,file_name], 'data')
    end
end