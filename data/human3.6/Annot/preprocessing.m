maindir = 'D:\dissertation\data\human3.6\Annot';
subdir  = dir( maindir );
count = 0;
for i = 1 : length( subdir )
    if( isequal( subdir( i ).name, '.' )||...
        isequal( subdir( i ).name, '..')||...
        ~subdir( i ).isdir)               % �������Ŀ¼������
        continue;
    end
    subdirpath = fullfile( maindir, subdir( i ).name, '*.cdf' );
    dat = dir( subdirpath )               % ���ļ������Һ�׺Ϊdat���ļ�

    for j = 1 : length( dat )
        %datpath = fullfile( maindir, subdir( i ).name, dat( j ).name);
        %disp(datpath)
        %data = cdfread(datpath)
        %filepath = strcat(datpath,".mat")
        %save(filepath, 'data')
        count = count + 1
        % �˴������Ķ��ļ���д���� %
    end
end
disp(count)