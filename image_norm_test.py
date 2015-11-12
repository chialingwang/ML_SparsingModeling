import os, sys
import os.path
import struct

pwd = os.pardir;
accessPath = r"%s\patch_database_double" %pwd;

'''
def read_single(sample, image , patchSize):
    result = []
    f = open(r'%s\sample%s_%s_%dx%d' %( accessPath ,str(sample).zfill(2) , str(image).zfill(3) , patchSize , patchSize), 'rb+')
    read_data = f.read();
    #for i in read_data:
    #    print(i);
    for i in range(0 , len(read_data) , 8):

        b = read_data[i:i+8]

        data = struct.unpack('f', b)
        result.append(data)

    return result
'''

def read_fileName(file , size):
    result = []
    f = open(r'%s\%s' %( accessPath , file), 'rb+')
    read_data = f.read();
    n = size*size
    #for i in read_data:
    #    print(i);
    count = 0
    temp = []
    for i in range(0 , len(read_data) , 8):
       
        b = read_data[i:i+8]
        data = struct.unpack('d', b)
        temp.append(data)
        count += 1
        if(count == n):
            temp = []
            result.append(temp)
            count = 0
    return result