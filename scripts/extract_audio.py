# coding: utf-8
# prepare C3D caffeList and save kmeans data
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen
import subprocess
import librosa
import librosa.display
from sklearn.cluster import KMeans
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import pdb 

def invert_to_sound (specname):
    specinfo = np.load(specname)
    original = utils.ispecgram(specinfo,
              n_fft=512,
              hop_length=None,
              mask=True,
              log_mag=True,
              re_im=False,
              dphase=True,
              mag_only=False,
              num_iters=1000)

    plt.plot(original)
    plt.show()
    # scale the original array to audible sound
    write(specname +'Inverse.wav', 16000, original)

def concat_to_spec(combine): 
    XX = np.array(np.vsplit(combine, 2))
    p = XX[0]
    m = XX[1]
    p = p[:, :, np.newaxis]
    m = m[:, :, np.newaxis]
    XXX = np.concatenate((m,p),axis = 2)
    print XXX.shape
    return XXX

def repeat_center(v):
    r = v
    for i in range(62):
        r = np.concatenate((r, v), axis = 1)
    return r
    # print r.shape
    # np.save(name + '1s.npy', r)

def center_to_spec(ori, centers, label):
    result = ori
    for index, value in enumerate(ori):
        label_index = label[index]
        c = centers[label_index]
        result[index] = c
    return result

# os.chdir("/home/yuanxin/Downloads/viola_data/mp4/")
# for file in sorted(glob.glob("*.MP4")):
#     command = "ffmpeg -i "+ file + " -ab 160k -ac 2 -ar 44100 -vn ../audio/" + file + ".wav"
#     # print command
#     # import pdb
#     # pdb.set_trace()
#     subprocess.call(command, shell=True)

print 'extract audio done!'

os.chdir("/home/yuanxin/Downloads/viola_data/audio")
kmeans_label_num = {}
cnt = 0
combineAll = []
dir = []
f = open('/home/yuanxin/code/bitplanes-tracking/caffe.list', 'w')
for file in sorted(glob.glob("*.wav")):
    if cnt == 9:
        break
    print file
    suffix = '.MP4.wav'
    if file.endswith(suffix):
       directory = file[:-len(suffix)]
    dir.append(directory)
    fname = file
    sr = 15360
    vr = 60
    length = 16
    audio = utils.load_audio(fname, sample_length=-1, sr=sr)
    sample_length = audio.shape[0]

    spec = utils.specgram(audio,
                n_fft=512,
                hop_length=None,
                mask=True,
                log_mag=True,
                re_im=False,
                dphase=True,
                mag_only=False)

#     # plt.figure()
    mag = spec[:,:,0]
    dphase = spec[:,:,1]
    combine = np.concatenate((dphase, mag), axis=0)
    combineT = np.transpose(combine)
    kmeans_label_num[directory] = combineT.shape[0]
    combineAll.append(combineT)
    cnt += 1    

cluster_num = 256
combineAll = np.concatenate(combineAll)
kmeans = KMeans(n_clusters= cluster_num, random_state=0).fit(combineAll)
# np.savetxt('1.txt', kmeans.labels_)
center = kmeans.cluster_centers_
np.save('/home/yuanxin/code/bitplanes-tracking/spec.npy',combineAll)
np.save('/home/yuanxin/code/bitplanes-tracking/center.npy',kmeans.cluster_centers_)
np.save('/home/yuanxin/code/bitplanes-tracking/label.npy',kmeans.labels_)
# plt.figure()
# plt.plot(kmeans.labels_)
# plt.show()
# pdb.set_trace()

offset = 0
chuck_length = 16
# create list according video
for subdir in dir:
    print subdir    
    os.chdir("/home/yuanxin/Downloads/viola_data/image/" + subdir)
    length_image = len(sorted(glob.glob("*.jpg")))
    cnt = 0
    # bias = 0
    for file in sorted(glob.glob("*.jpg")):
        # dir name + start frame + label
        if cnt > length_image - chuck_length:
            break
        # chose rainbow_gram sample center as label
        cluster_label = kmeans.labels_[offset + cnt + chuck_length/2 -1]
        tmp = subdir + ' ' + str(cnt+1) + ' ' + str(cluster_label) + '\n'
        f.write(tmp)
        cnt += 1
        
    offset += kmeans_label_num[subdir]