# coding: utf-8
import sys
import glob, os
import numpy as np
import array
import matplotlib.pyplot as plt
from magenta.models.nsynth import utils
import subprocess
import librosa
import librosa.display
from sklearn.cluster import KMeans
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

def main():
    plt.figure()
    video_index = '3229'
    root_folder = '/home/yuanxin/Downloads/viola_data/baseline_result/'
    os.chdir(root_folder)
    origin_name = './Origin_wave/' + video_index + '.wav'
    vgg_name = './VGG_wave/' + video_index + '.wav'
    gan_name = './GAN_wave/' + video_index + '.wav'

    sr = 16000
    cnt = 0
    audio_name_all = [origin_name,vgg_name,gan_name]
    for i in audio_name_all:
        audio = utils.load_audio(i, 5*sr, sr=sr)
        cnt += 1
        plt.subplot(6,1,cnt)
        plt.plot(audio)
        spec = utils.specgram(audio,
                    n_fft=512,
                    hop_length=None,
                    mask=True,
                    log_mag=True,
                    re_im=False,
                    dphase=True,
                    mag_only=False)
        dphase = spec[:,:,1]
        cnt += 1
        plt.subplot(6,1,cnt)
        librosa.display.specshow(dphase)
    plt.show()

    
if __name__ == "__main__":
    main()


