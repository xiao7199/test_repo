import numpy as np
# import skvideo.io

# dirty usage 
# for list create
prefix = 'output/test_3229/'

f = open('./output_prefix.list', 'w')
for i in range(2224):
    tmp = prefix +  '%06d' % (i+1) + '\n'
    print tmp
    f.write(tmp)
f.close()

f = open('./test_3229.lst', 'w')
for i in range(2224):
    tmp = '/home/rll/Desktop/yuanxin/caffe_dirs/C3D/C3D-v1.0/examples/c3d_feature_extraction/viola_input/GOPR3229/ ' +  str(i+1) + ' 0' + '\n'
    print tmp
    f.write(tmp)
f.close()
    
    

