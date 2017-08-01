import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import pdb

class c3d_encoder_model(nn.Module):
	def __init__(self, predict_class):
		super(c3d_encoder_model,self).__init__()
		self.conv_encoder = nn.Sequential(
			nn.Conv3d(3, 64, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.MaxPool3d((2,2,1)),
			nn.Conv3d(64, 128, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.MaxPool3d((2,2,2)),
			nn.Conv3d(128, 256, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Conv3d(256, 256, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.MaxPool3d((2,2,2)),
			nn.Conv3d(256, 512, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Conv3d(512, 512, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.MaxPool3d((2,2,2)),
			nn.Conv3d(512, 512, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Conv3d(512, 512, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.MaxPool3d((2,2,2))
			)
		self.fc_layer = nn.Sequential(
			nn.Linear(81920, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, predict_class),
			nn.LogSoftmax()
			)
	def forward(self, data, get_conv_feature = False):
		conv_feature =  self.conv_encoder(data)
		if get_conv_feature is True:
			return self.fc_layer(conv_feature.view(data.size(0),-1)), conv_feature
		return self.fc_layer(conv_feature.view(data.size(0),-1))

class c3d_decoder_model(nn.Module):
	def __init__(self):
		super(c3d_decoder_model,self).__init__()
		self.conv_decoder = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv3d(512, 512, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Conv3d(512, 512, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv3d(512, 512, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Conv3d(512, 256, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv3d(256, 256, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Conv3d(256, 128, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv3d(128, 64, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv3d(64, 3, (3,3,3), stride = (1,1,1), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			)
	def forward(self, conv_feature):
		return self.conv_decoder(conv_feature)

class shift2rg_generator(nn.Module):
	def __init__(self):
		super(softmax2rg_generator,self).__init__()
		self.generator = nn.Sequential(
			nn.Linear(81920, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, 514)
			)
	def forward(self, conv_feature, kmean_center):
		return (1+self.generator(prob))*kmean_center

class softmax2rg_discriminator(nn.Module):
	def __init__(self):
		super(softmax2rg_discriminator,self).__init__()
		self.discriminator = nn.Sequential(
			nn.Linear(514, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 1),
			nn.Sigmoid()
			)
	def forward(self, rg):
		return self.discriminator(rg)