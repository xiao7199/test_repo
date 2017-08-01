from model import *
from utils import *
import numpy as np
import torch,pdb
import torch.nn as nn
import time

param = {}
param['display_interval'] = 100
param['test_interval'] = 500
param['batch_size'] = 20
param['epoch'] = 20
param['img_data_path'] = ''
param['gt_txt_path'] = ''
param['data_folder_name'] = ['3229','3230','3231','3232','3233','3234','3235','3236','3237']
param['test_data_folder_name'] = ['']
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

FCN_model = FCN32s(5)
SGD_optimizer = torch.optim.SGD(FCN_model.parameters(), lr = param['lr_rate'], \
					momentum = param['momentum'], weight_decay = param['weight_decay'])
loss_fn = torch.nn.BCELoss()

loss_fn.cuda()
FCN_model.cuda()
SGD_optimizer.cuda()
scheduler = torch.optim.lr_scheduler.StepLR(SGD_optimizer, param['step_size'], param['gamma'])
scheduler.cuda()

counter = 0 
loss_record = 0
start_time = time.time()


t_dataloader = Dataloader(param, mode = 'test')
test_dataloader = torch.utils.data.DataLoader(dataset = t_dataloader,
										   batch_size=param['batch_size'], 
										   shuffle=False,
										   num_workers=4)
test_data_iter = iter(test_dataloader)
for train_data, train_gt in data_iter:
	
	if train_data is None:
		continue

	train_data = Variable(train_data.cuda())
	gt_rb_vector = Variable(train_gt.cuda())

	FCN_model.zero_grad()

	pred_data = FCN_model(train_data)

	loss = loss_fn(pred_data, train_gt)
	loss.backward()
	loss_record += np.mean(loss.data.cpu())
	scheduler.step()

	counter += 1

	if counter % param['display_interval'] == 0 and counter != 0:
		print('iter :{}, train_loss:{}, time:{}'.format(counter, loss_record/param['display_interval'],\
						 time.time() - start_time))
		start_time = time.time()
		loss_record = 0

	if counter % param['test_interval'] == 0 and counter != 0:
		print('start testing')
		test_loss = 0
		test_counter = 0
		for test_data, test_gt in test_data_iter:
			test_data = Variable(test_data.cuda())
			test_gt = Variable(test_gt.cuda())
			pred_data = FCN_model(test_data)
			loss = loss_fn(pred_data, train_gt)
			test_loss += np.mean(loss.data.cpu())
			test_counter += 1
		print('Done testing, testing error is {}'.format(test_loss/test_counter))