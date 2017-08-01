from model import *
from utils import *
import numpy as np
import torch,pdb
import torch.nn as nn


param = {}
param['mode'] = 'train'
param['chunk_size'] = 16
param['batch_size'] = 16
param['window_size'] = 16
sparam['kmeans_num'] = 256
param['epoch'] = 20
param['data_path'] = ''
param['audio_path'] = '/home/yuanxin/Downloads/viola_data/audio/'
param['data_folder_name'] = ['3229','3230','3231','3232','3233','3234','3235','3236','3237']

param['img_h'] =  120
param['img_w'] =  160
param['mean_val'] = np.zeros((128,128,128))
param['lr_rate'] = 0.0001
param['momentum'] = 0.9
param['weight_decay'] = 0.0005
param['step_size'] = 5000
param['gamma'] = 0.1

dataloader = Dataloader(param)
train_loader = torch.utils.data.DataLoader(dataset = dataloader,
										   batch_size=param['batch_size'], 
										   shuffle=True,
										   num_workers=4)
data_iter = iter(train_loader)

MODEL_c3d_e = c3d_encoder_model(param['kmeans_num'])
MODEL_shift2rg_g = shift2rg_generator()
SGD_optimizer = torch.optim.SGD(MODEL_c3d_e.parameters()+MODEL_shift2rg_g.parameters(), lr = param['lr_rate'], \
					momentum = param['momentum'], weight_decay = param['weight_decay'])
entropy_loss_fn = torch.nn.NLLLoss()
l1_loss_fn = torch.nn.SmoothL1Loss()

l1_loss_fn.cuda()
entropy_loss_fn.cuda()
MODEL_c3d_e.cuda()
MODEL_shift2rg_g.cuda()
SGD_optimizer.cuda()
scheduler = torch.optim.lr_scheduler.StepLR(SGD_optimizer, param['step_size'], param['gamma'])
scheduler.cuda()

for train_data, gt_rb_idx, gt_rb_vector, gt_kmeans_center in data_iter:
	
	train_data = Variable(train_data.cuda())
	gt_rb_vector = Variable(gt_rb_vector.cuda())
	gt_rb_idx = Variable(gt_rb_idx.cuda())
	gt_kmeans_center = Variable(gt_rb_idx.cuda())

	MODEL_c3d_e.zero_grad()
	MODEL_shift2rg_g.zero_grad()

	sfmax_prob, conv_feature = MODEL_c3d_e(train_data, get_conv_feature = True)
	rb_pred = MODEL_shift2rg_g(conv_feature, gt_kmeans_center)

	l1_loss = l1_loss_fn(rb_pred, gt_rb_vector)
	l1_loss.backward()
	entropy_loss = entropy_loss_fn(sfmax_prob, gt_rb_idx)
	entropy_loss.backward()
	scheduler.step()
