maindir = 'D:\dissertation\data\human3.6\H36M-images\images';
subdir  = dir( maindir );
count = 0;
for i = 1 : length( subdir )
    if( isequal( subdir( i ).name, '.' )||...
        isequal( subdir( i ).name, '..')||...
        ~subdir( i ).isdir)               % 如果不是目录则跳过
        continue;
    end
    subdirpath = fullfile( maindir, subdir( i ).name, '*.jpg' );
    dat = dir( subdirpath )               % 子文件夹下找后缀为dat的文件

    for j = 1 : length( dat )
        count = count + 1;
        % 此处添加你的对文件读写操作 %
    end
end
disp(count)