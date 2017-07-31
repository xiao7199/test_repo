import sys
import glob, os
# sys.path.insert(0, os.path.join('../caffe', 'python'))
# import caffe
import numpy as np
import matplotlib.pyplot as plt
import array
from sklearn import preprocessing

folder = ['test_3229','test_3230','test_3231','test_3232','test_3233','test_3234','test_3235','test_3236','test_3237']
feat_name = ['label_3229','label_3230','label_3231','label_3232','label_3233','label_3234','label_3235','label_3236','label_3237']
root_folder = '/home/yuanxin/Downloads/output/'

for sub_folder,save_feat_name in zip(folder,feat_name):
    os.chdir(root_folder + sub_folder)
    feat_map = []
    for file in sorted(glob.glob("*.prob")):
        # print file
        f = open(file, "rb")
        size = array.array("i")  # L is the typecode for uint32
        size.fromfile(f, 5)
        f.seek(20, os.SEEK_SET)
        total = size[0]*size[1]*size[2]*size[3]*size[4]
        data = array.array("f")
        data.fromfile(f,total)
        data = np.array(data)
        data = data.reshape(size[0],size[1],size[2],size[3],size[4])
        # print data.shape
        feat_map.append(data)
        f.close()
    feat_map = np.array(feat_map)
    print feat_map.shape
    np.save('/home/yuanxin/code/bitplanes-tracking/gan_input/' + save_feat_name, feat_map)

# max_index = []
# for i in range(feat_map.shape[0]):
#     max_index.append(np.argmax(feat_map[i][0].reshape(16)))
# np.save('/home/yuanxin/code/bitplanes-tracking/label_3238', max_index)