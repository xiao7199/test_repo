import cv2
import glob, os
import numpy as np
import skvideo.io

writer_lefthand = skvideo.io.FFmpegWriter("output_lefthand.mp4", outputdict={
  '-vcodec': 'libx264', '-b': '300000000'
})

os.chdir("./imgs")
cnt = 0
inter = 500
for file in sorted(glob.glob("*.jpg")):
    print file
    mask_file = '../img1.bmp'
    mask = cv2.imread(mask_file)
    mask_hand = np.array(mask, copy=True)  
    origin = cv2.imread(file)
    left_hand = np.multiply(origin, mask_hand/255.0)
    cv2.imwrite('../mask_hand/' + file, left_hand)

    # change RGB channel
    left_hand_video=np.array(left_hand,copy = True)
    left_hand_video[:,:,0]=left_hand[:,:,2]
    left_hand_video[:,:,1]=left_hand[:,:,1]
    left_hand_video[:,:,2]=left_hand[:,:,0]
    writer_lefthand.writeFrame(left_hand_video)
    cnt += 1

writer_lefthand.close()