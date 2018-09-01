maindir = 'D:\dissertation\data\human3.6\Annot';
subdir  = dir( maindir );
count = 0;
for i = 1 : length( subdir )
    if( isequal( subdir( i ).name, '.' )||...
        isequal( subdir( i ).name, '..')||...
        ~subdir( i ).isdir)               % 如果不是目录则跳过
        continue;
    end
    subdirpath = fullfile( maindir, subdir( i ).name, '*.cdf' );
    dat = dir( subdirpath )               % 子文件夹下找后缀为dat的文件

    for j = 1 : length( dat )
        %datpath = fullfile( maindir, subdir( i ).name, dat( j ).name);
        %disp(datpath)
        %data = cdfread(datpath)
        %filepath = strcat(datpath,".mat")
        %save(filepath, 'data')
        count = count + 1
        % 此处添加你的对文件读写操作 %
    end
end
disp(count)