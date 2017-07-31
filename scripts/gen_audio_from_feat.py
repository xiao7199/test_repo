# coding: utf-8
import sys
import glob, os
import numpy as np
import array
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

# ******* Main Function *******
# Input: 
# - specname: string name of Transpose Specgram : N * 514
# - centernum: number of kmeans center
# - savename: name of output KMeans Transpose Specgram
# Output: Transpose KMeans Spec : N * 514
# def KMeansSpec(specname, centernum, savename):
#     fname = specname + '.npy'
#     X = np.load(fname)
#     kmeans = KMeans(n_clusters= centernum, random_state=0).fit(X)
#     X_label = kmeans.labels_
#     center = kmeans.cluster_centers_
#     Y = center_to_spec(X, center, X_label)
#     np.save(savename + '.npy', Y)
    return Y

def main():
    video_all = ['3229','3230','3231','3232','3233','3234','3235','3236','3237']
    save_folder = '/home/yuanxin/code/bitplanes-tracking/'
    
    for video_index in video_all:
        data_folder = '/home/yuanxin/Downloads/output/test_' + video_index
        specname = './spec.npy'
        centername = './center.npy'
        os.chdir(data_folder)
        feat_map = []
        for file in sorted(glob.glob("*.prob")):
            print file
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

        max_index = []
        for i in range(feat_map.shape[0]):
            max_index.append(np.argmax(feat_map[i][0].reshape(16)))
        
        os.chdir(save_folder)
        audioname = './VGG_out/inverse_' + video_index + '.wav'
        X = np.load(specname)
        center = np.load(centername)
        label = max_index
        # print X.shape
        # print center.shape
        # print label.shape
        # import pdb
        # pdb.set_trace()
        Y = center_to_spec(X[0:len(label)], center, label)
        LabelT = np.transpose(Y)
        print 'LabelT', LabelT.shape
        # reshape to the correct rainbowgram format
        XXX = concat_to_spec(LabelT)
        print 'XXX', XXX.shape
        ori = utils.ispecgram(XXX,
                n_fft=512,
                hop_length=None,
                mask=True,
                log_mag=True,
                re_im=False,
                dphase=True,
                mag_only=False,
                num_iters=1000)
        # plt.plot(ori)
        # plt.show()
        # scale the original array to audible sound
        write(audioname, 16000, ori)

if __name__ == "__main__":
    main()


