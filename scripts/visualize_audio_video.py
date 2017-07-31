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

def main():
    root_folder = '/home/yuanxin/Downloads/viola_data/baseline_result/'
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


