import numpy as np
import torch.utils.data as data
import glob,os,pdb
import cv2
# import skimage.misc
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal

def get_gaussian_gt_label(label, sigma, img_h, img_w):
	label_num = label.shape[0]
	x_space = np.linspace(0, 1, img_w, endpoint=False)
	y_space = np.linspace(0, 1, img_h, endpoint=False)
	pos = np.zeros((img_h,img_w,2))
	pos[:,:,0] = x_space
	pos[:,:,1] = y_space
	gt_label = np.zeros((label_num,img_h,img_w))
	for i in range(label_num):
		rv = multivariate_normal(gt_label[i,:], [[sigma, 0], [0, sigma]])
		gt_label[i,:,:] = rv.pdf(pos)
	return gt_label

def read_gt_txt(filename):
	f = open(filename)
	key_point = []
	for (iii, line) in enumerate(f):
		if iii == 0:
			bbox_cnt = int(line.strip())
			continue
		tmp = [int(t.strip()) for t in line.split()]
		key_point.append(tmp)
	if len(key_point) != 5:
		return -1
	gt_label = np.zeros((5,2))
	for i in range(5):
		gt_label[i,:] = key_point[i]
	return gt_label

class Dataloader(data.Dataset):
	
	def __init__(self, param, mode = 'train'):


		self.data_folder = param['data_folder_name']
		if mode == 'test':
			self.folder_num = param['test_data_folder_name']
		self.folder_num = len(self.data_folder)
		# self.label_data_folder = param['label_data_folder']
		self.data_index_binbook = np.zeros((self.folder_num+1,1))
		self.rb_vector = []
		self.epoch = param['epoch']
		self.img_data_path = param['img_data_path']
		self.gt_txt_path = param['gt_txt_path']
		self.img_h = param['img_h']
		self.img_w = param['img_w']
		self.mean_val = param['mean_val']
		self.sigma = param['sigma']
		self.img_name_list = []
		temp_index = 0
		for index, data_folder_name in enumerate(self.data_folder):
			print index
			self.img_name_list.append(glob.glob(os.path.join\
				(self.img_data_path,data_folder_name,'*.jpg')))
			curr_visual_size =  len(self.img_name_list[index])
			temp_index += curr_visual_size
			self.data_index_binbook[index+1] += temp_index
		self.total_load_num = temp_index
		print('finish initializing data loader')
	
	def __getitem__(self, index):

		data_index_in_epoch = index % self.total_load_num
		data_folder_bin_num = np.digitize(data_index_in_epoch, self.data_index_binbook)
		data_folder_name = self.data_folder[data_folder_bin_num - 1]
		data_start_index_in_folder = data_index_in_epoch - self.data_index_binbook[data_folder_bin_num-1]
		img_name = self.img_name_list[index][data_start_index_in_folder]
		img_txt_label_name = img_name[:img_name.index('.')] + '.txt'

		gt_label = read_gt_txt(os.path.join(self.gt_txt_path, data_folder,img_txt_label_name))

		if gt_label == -1:
			return None,None
		
		img = cv2.imread(os.path.join\
			(self.img_data_path, data_folder,img_name )) - self.mean_val
		gaussian_map = get_gaussian_gt_label(gt_label, self.sigma, self.img_h, self.img_w)
		return np.transpose(img,[2,0,1]),gaussian_map

	def __len__(self):
		return self.total_load_num* self.epoch
