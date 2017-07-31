import numpy as np
import os
import glob
import cv2
import pdb
import skvideo.io

source_root_folder = '/home/yuanxin/Downloads/viola_data/mp4'
dest_root_folder = '/home/yuanxin/Downloads/viola_detector/labels_for_viola'
sub_dir = ['3275','3276','3277','3278','3279','3280','3281', '3282',
            '3283','3284','3285','3286','3287','3288','3289','3290','3291']

# cd source then extract frames
for index in sub_dir:
    print index
    os.chdir(source_root_folder)
    inputparameters = {}
    outputparameters = {}
    reader = skvideo.io.FFmpegReader('GOPR' + index + '.MP4',
                    inputdict=inputparameters,
                    outputdict=outputparameters)
    # create dest directory
    if not os.path.exists(dest_root_folder + '/' + 'Images/' + index):
        os.makedirs(dest_root_folder + '/' + 'Images/' + index)
    if not os.path.exists(dest_root_folder + '/' + 'Masks/' + index):
        os.makedirs(dest_root_folder + '/' + 'Masks/' + index)
    if not os.path.exists(dest_root_folder + '/' + 'Labels/' + index):
        os.makedirs(dest_root_folder + '/' + 'Labels/' + index)

    os.chdir(dest_root_folder + '/' + 'Images/' + index)
    cnt = 0
    for frame in reader.nextFrame():
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (640, 480))
        cv2.imwrite(str(cnt) + '.jpg', frame)
        cnt += 1