import os, sys
import os.path
import struct

pwd = os.pardir;
accessPath = r"%s\patch_database_double" %pwd;

def read(sample, image , patchSize):
    result = []
    f = open(r'%s\sample%s_%s_%dx%d' %( accessPath ,str(sample).zfill(2) , str(image).zfill(3) , patchSize , patchSize), 'rb+')
    read_data = f.read();
    #for i in read_data:
    #    print(i);
    for i in range(0 , len(read_data) , 4):

        b = read_data[i:i+4]

        data = struct.unpack('f', b)
        result.append(data)

    return result