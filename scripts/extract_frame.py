import cv2
import glob, os
import numpy as np
import skvideo.io

# here you can set keys and values for parameters in ffmpeg
extract = False

frame = 0
if(extract):
    inputparameters = {}
    outputparameters = {}
    reader = skvideo.io.FFmpegReader('./GOPR3241.MP4',
                    inputdict=inputparameters,
                    outputdict=outputparameters)

    cnt = 0
    for frame in reader.nextFrame():
        if (cnt % 100 == 0):
            print cnt
            # roi = frame[r:r+h, c:c+w]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480)) 
            cv2.imwrite('../BBox-Label-Tool/Images/002/' + str(cnt) + '.jpg', frame)
        cnt += 1
else:
    label_dir = '../BBox-Label-Tool/Labels/002/'
    list = []
    for i in range(11):
        with open(label_dir + str(i*100) + '.txt') as f:
            for (i, line) in enumerate(f):
                if i == 0:
                    bbox_cnt = int(line.strip())
                    continue
                tmp = [int(t.strip()) for t in line.split()]
                # print tmp
                list.append(tuple(tmp))
        
    print list    
        # import pdb
        # pdb.set_trace()
    rows = 480
    cols = 640

    mask_file = './viola_mask_scale.png'
    # mask_file = '../viola_mask_scale.png'
    mask = cv2.imread(mask_file)
    mask = cv2.resize(mask, (640, 480)) 
    for index in range(0,len(list),2):
        print index
        viola_box = list[index]
        hand_box = list[index+1]
        for i in range(mask.shape[2]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[0]):
                    if(((viola_box[1]< k < viola_box[3]) & (viola_box[0] < j < viola_box[2])) or ((hand_box[1] < k < hand_box[3]) & (hand_box[0] < j < hand_box[2]))):
                        mask[k,j,i] = 255
                    else:
                        mask[k,j,i]  = 0
        mask_ori = cv2.resize(mask, (1280, 960)) 
        cv2.imwrite('mask_xin_' + str(index/2*100) +".jpg",mask_ori)
            
            
            
        


        



