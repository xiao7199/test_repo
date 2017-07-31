import cv2
import glob, os
import numpy as np
import skvideo.io

writer_violin = skvideo.io.FFmpegWriter("violin_stable.mp4", outputdict={
  '-vcodec': 'libx264', '-r': '60'
})

os.chdir("./data_1/imgs")
for file in sorted(glob.glob("*.jpg")):
    print file
    violin = cv2.imread(file)
    violin_video=np.array(violin,copy = True)
    violin_video[:,:,0]=violin[:,:,2]
    violin_video[:,:,1]=violin[:,:,1]
    violin_video[:,:,2]=violin[:,:,0]
    writer_violin.writeFrame(violin_video)


writer_violin.close()