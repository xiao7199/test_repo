import numpy as np
# import skvideo.io

# dirty usage for list create

# os.chdir("./mask_hand")
# subdir_name = '3229/'
# cnt = 0
# inter = 16
# f = open('../train.list', 'w')
# for file in sorted(glob.glob("*.jpg")):
#     # print file
#     if cnt % 16 == 0:
#         f.write(subdir_name + ' ' +  str(cnt) + ' ' + str(0) + '\n')
#     cnt += 1

# f.close()

prefix = 'output/test_3237/'

f = open('./output_prefix.list', 'w')
for i in range(2101):
    tmp = prefix +  '%06d' % (i+1) + '\n'
    print tmp
    f.write(tmp)
f.close()

f = open('./test_3238.lst', 'w')
for i in range(2224):
    tmp = '/home/rll/Desktop/yuanxin/caffe_dirs/C3D/C3D-v1.0/examples/c3d_feature_extraction/viola_input/GOPR3238/ ' +  str(i+1) + ' 0' + '\n'
    print tmp
    f.write(tmp)
f.close()
    
    

