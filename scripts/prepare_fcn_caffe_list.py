import numpy as np
import os
import glob
import cv2
import pdb
import skimage.io

all_folder = ['3286','3289','3229','3230','3231','3232','3233','3234','3235','3236','3237','3238','3239','3240','3241']
root_folder = '/home/yuanxin/Downloads/viola_detector/labels_for_viola'
color_table = [(1,1,1),(2,2,2),(3,3,3),(4,4,4),(5,5,5)]
ff = open('/home/yuanxin/Desktop/FCN.lst', 'w')

# prepare caffe train list
for i in all_folder:
    print i
    os.chdir(root_folder + '/Labels/' + i)
    for labelfilename in sorted(glob.glob("*.txt")):
        # remove suffix
        filename = labelfilename[0:-4]
        # print filename
        # pdb.set_trace()        
        bbox_cnt = 0
        key_point = []
        if os.path.exists(labelfilename):
            with open(labelfilename) as f:
                for (iii, line) in enumerate(f):
                    if iii == 0:
                        bbox_cnt = int(line.strip())
                        continue
                    tmp = [int(t.strip()) for t in line.split()]
                    key_point.append(tmp)
        if len(key_point) != 5:
            continue
        # get y coord
        y_min_index = np.argmin([i_tmp[1] for i_tmp in key_point])
        # print y_min_index
        sorted_key_point = []
        sorted_key_point.append(key_point[y_min_index])
        # remove top point
        del key_point[y_min_index]
        # get x
        x_pt = [i_x[0] for i_x in key_point]
        y_pt = [i_y[1] for i_y in key_point]
        x_idx = sorted(range(len(x_pt)), key=lambda k: x_pt[k])
        # print x_idx
        if y_pt[x_idx[0]] < y_pt[x_idx[1]]:
            sorted_key_point.append(key_point[x_idx[0]])
            sorted_key_point.append(key_point[x_idx[1]])
        else:
            sorted_key_point.append(key_point[x_idx[1]])
            sorted_key_point.append(key_point[x_idx[0]])
        
        if y_pt[x_idx[2]] < y_pt[x_idx[3]]:
            sorted_key_point.append(key_point[x_idx[3]])
            sorted_key_point.append(key_point[x_idx[2]])
        else:
            sorted_key_point.append(key_point[x_idx[2]])
            sorted_key_point.append(key_point[x_idx[3]])

        # print sorted_key_point

        #draw circle
        
        # Create a black image
        if(i == '3286' or i == '3289'):
            img = np.zeros((600,800), np.uint8)
        else:
            img = np.zeros((480,640), np.uint8)
        for ii in range(5):
            # Draw a red closed circle
            cv2.circle(img,(sorted_key_point[ii][0],sorted_key_point[ii][1]), 20, color_table[ii], -1)
            #Display the image
        img = cv2.resize(img, (1280, 960)) 
        save_path = '/Masks/' + i + '/'
        mask_name = save_path + filename + '.png'
        ori_path =  '/Images/' + i + '/'
        ori_name = ori_path + filename + '.jpg'
        
        skimage.io.imsave(root_folder + mask_name, img)
        line = ori_name + ' ' + mask_name + '\n'
        ff.write(line)

ff.close()
        
