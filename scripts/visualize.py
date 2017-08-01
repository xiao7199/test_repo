import sys
import os
# sys.path.insert(0, os.path.join('../caffe', 'python'))
# import caffe
import numpy as np
import matplotlib.pyplot as plt
import cv2
import array
from sklearn import preprocessing

feat_map = []

clip_name = '000145.'
# clip_name = '000001.'
# clip_name = '000017.'
# clip_name = '000033.'
# clip_name = '000049.'
# clip_name = '000065.'
# clip_name = '000081.'
# clip_name = '000113.'
# clip_name = '000129.'
data_folder = './c3d/violin/'


clip_name = data_folder + clip_name
type_name = ['data','conv1a','conv2a','conv3a','conv4b','conv5b']
name_array = [clip_name + i for i in type_name]

for name in name_array:
    f = open(name, "rb")
    size = array.array("i")  # L is the typecode for uint32
    size.fromfile(f, 5)
    f.seek(20, os.SEEK_SET)
    total = size[0]*size[1]*size[2]*size[3]*size[4]
    data = array.array("f")
    data.fromfile(f,total)
    data = np.array(data)
    data = data.reshape(size[0],size[1],size[2],size[3],size[4])
    feat_map.append (data[0])
    f.close()

print len(feat_map)
show_blob = feat_map



# img = np.transpose(img,(1,2,3,0))
# print img.shape
# data = img[1]
# print data.shape
# plt.subplot(1,1,1)
# plt.imshow(data[:,:,1],cmap=plt.cm.gray)
# plt.show()

fig = plt.figure(figsize=(20,20))
feat_name = ['frames','conv1a','conv2a','conv3a','conv4b','conv5b']

#for conv1
for num_feat in range(6):
    feat_map = show_blob[num_feat]
    max_conv = np.zeros((feat_map.shape[0],feat_map.shape[1]))

    print feat_map.shape
    for i in xrange(feat_map.shape[0]):
        for j in xrange(feat_map.shape[1]):
            max_conv[i,j] = np.sum(feat_map[i,j,:,:])

    index = np.argmax(max_conv, axis=0)
    feat_map = feat_map[index[0],:,:,:].reshape(1,feat_map.shape[1],
        feat_map.shape[2],feat_map.shape[3])
    feat_map = np.transpose(feat_map,(1,2,3,0))
    for i in range(len(feat_map)):
        for j in range(1):
            #clip_size:16, feat_num:6
            ax = fig.add_subplot(6,16, num_feat * 16 + i + 1)
            if i == 0:
                ax.annotate(feat_name[num_feat],xy=(0.5, 0.5),xytext=(-ax.yaxis.labelpad - 22, 0)
                ,xycoords='axes fraction',textcoords='offset points',
                size='small', ha='right', va='center')
            data = feat_map[i]
            data = data[:,:,j]
            # data = (data - data.min()) / (data.max() - data.min())
            data = preprocessing.scale(data)
            if num_feat == 0:
                color = plt.cm.gray
            else:
                color = plt.cm.jet
            plt.imshow(data,cmap=color)
            plt.axis('off')

            plt.subplots_adjust(top = 0.98,bottom=0,left=0.04,right=0.97,wspace = 0.11,hspace = 0.08)
# plt.tight_layout()
plt.suptitle('Violin C3D_Feature Visualization',fontsize=8)

# fig.set_ylabel(['A','B','C','D','E','F'])
plt.show()
