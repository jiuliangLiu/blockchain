# -*- coding: utf-8 -*-
import re
#base_read_path='G:\Python_Crawler\coverContracts\coverContract{}.sol'
base_read_path='G:\coverContract\coverContract(no annotation)\coverContract{}.sol'
#base_write_path='G:\Python_Crawler\coverContracts(no_annotation)\coverContract{}.sol'
base_write_path='G:\Python_Crawler\coverContracts(no_annotation)\coverContract{}.sol'
for x in range(15000,15101):
    read_path=base_read_path.format(x)
    write_path=base_write_path.format(x+4962)
#    print(read_path)
    with open(read_path,'r',encoding='utf-8') as fp:
        source_code=fp.read()
#        print(source_code)
#        break;
        pattern1=re.compile(r'//.*')
        pattern2=re.compile(r'/\*\*?.*?\*/',re.S)
        pattern3=re.compile(r'[\r\n]{2,}')
#        matchNotation=re.findall(r'(//.*)|(/\*\*[.\n]*\*/)',source_code)
        
        matchNotation1=pattern1.findall(source_code)
#        print(matchNotation1[0])
#        source_code=source_code.replace(matchNotation1[0],'')
#        print(source_code)
        for index,x in enumerate(matchNotation1):
            source_code=source_code.replace(x,"",1)
#            print(x)
        matchNotation2=pattern2.findall(source_code)    
        for index,x in enumerate(matchNotation2):
            source_code=source_code.replace(x,'',1)
#            print(x)
        matchNotation3=pattern3.findall(source_code)
        for index,x in enumerate(matchNotation3):
            source_code=source_code.replace(x,'\n',1)
#            print('第',index,'个元素为：',x)
#        print(source_code)
#        print(matchNotation1)
#        print(matchNotation2)
#        print(matchNotation3)
        with open(write_path,'w',encoding='utf-8') as fp:
            fp.write(source_code)
#        break   
