import numpy as np
import torch.utils.data as data
import glob,os,pdb
import cv2
import skimage.misc
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen
import subprocess
import librosa
import librosa.display
from scipy.io.wavfile import write
from sklearn.cluster import KMeans

def get_rb_vector(fname, sr = 15360, window_size = 16):
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
	mag = spec[:,:,0]
	dphase = spec[:,:,1]
	rb_vector = np.concatenate((dphase, mag), axis=0)
	return rb_vector[window_size/2:,:]

def get_rb_kmeans_center(rb_data, kmeans_center_num):
	kmeans = sklearn.cluster.KMeans(n_clusters = kmeans_center_num).fix(rb_data)
	return kmeans.labels_, kmeans.cluster_centers_

class Dataloader(data.Dataset):
	
	def __init__(self, param):

		self.folder_num = len(param['data_folder_name'])
		self.data_folder = param['data_folder_name']
		self.data_index_binbook = np.zeros((self.folder_num+1,1))
		self.rb_vector = []
		self.audio_path = param['audio_path']
		self.epoch = param['epoch']
		self.window_size = param['window_size']
		self.data_path = param['data_path']
		self.batch_size = param['batch_size']
		self.img_h = param['img_h']
		self.img_w = param['img_w']
		self.chunk_size = param['chunk_size'] 
		self.mean_val = param['mean_val']

		temp_index = 0
		for index, data_folder_name in enumerate(data_folder):
			curr_visual_size =  len(glob.glob(os.path.join\
				(data_path,data_folder_name,'*.jpg'))) - window_size + 1
			rb_data = get_rb_vector(self.audio_path + 'GOPR' + data_folder_name + '.MP4.wav')
			curr_size = np.min(curr_visual_size, rb_data.shape[0])
			temp_index += curr_size
			self.data_index_binbook[index+1] += temp_index			
			self.rb_vector.append(rb_data[:curr_size,:])
		self.rb_vector = np.concatenate(self.rb_vector, axis = 0)
		self.total_load_num = temp_index
		self.rb_size = rb_data.shape[1]
		self.rb_kmeans_label, self.rb_kmeans_center = get_rb_kmeans_center(self.rb_vector, param['kmeans_num'])
		print('finish initializing data loader')
		
	def get_kmeans_center():
		return self.rb_kmeans_center
	
	def __getitem__(self, index):

		data_index_in_epoch = index % self.total_load_num
		data_folder_bin_num = np.digitize(data_index_in_epoch, self.data_index_binbook)
		data_folder_name = self.data_folder[data_folder_bin_num - 1]
		data_start_index_in_folder = data_index_in_epoch - self.data_index_binbook[data_folder_bin_num-1]

		rb_data = self.rb_vector[data_folder_bin_num : self.batch_size + data_folder_bin_num]
		rb_center_label = self.rb_kmeans_label[data_folder_bin_num : self.batch_size + data_folder_bin_num]
		rb_kmeans_data = self.rb_kmeans_center[rb_center_label,:]
		image_chunk_data = np.zeros((self.chunk_size, self.img_h, self.img_w,3))
		for i in range(self.chunk_size):
			 img = cv2.imread(os.path.join\
				(self.data_path, data_folder, str(data_start_index_in_folder+i)+'.jpg')) - self.mean_val
			 image_chunk_data[i,:,:,:] = scipy.misc.imresize(img, [self.img_h, self.img_w])
		return np.transpose(image_chunk_data,[3,0,1,2]), rb_center_label, rb_data, rb_kmeans_data

	def __len__(self):
		return self.total_load_num* self.epoch
