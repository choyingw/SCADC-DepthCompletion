import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import cv2
import collections
import matplotlib.pyplot as plt
from .submodule import *

def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'lambda':
		lambda_rule = lambda epoch: opt.lr_gamma ** ((epoch+1) // opt.lr_decay_epochs)
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer,step_size=opt.lr_decay_iters, gamma=0.1)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler


def init_weights(net, init_type='normal', gain=0.02):
	net = net
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=gain)
			elif init_type == 'pretrained':
				pass
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None and init_type != 'pretrained':
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)
	print('initialize network with %s' % init_type)
	net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)

	for root_child in net.children():
		for children in root_child.children():
			if children in root_child.need_initialization:
				init_weights(children, init_type, gain=init_gain)
	return net

def define_SCADCNet(init_type='xavier', init_gain=0.02, gpu_ids=[]):
	net = SCADCNetGenerator()
	return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################
class SAConv(nn.Module):
	# Convolution layer for sparse data
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
		super(SAConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
		self.if_bias = bias
		if self.if_bias:
			self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
		self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)
		nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
		self.pool.require_grad = False

	def forward(self, input):
		x, m = input
		x = x * m
		x = self.conv(x)
		weights = torch.ones(torch.Size([1, 1, 3, 3])).cuda()
		mc = F.conv2d(m, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
		mc = torch.clamp(mc, min=1e-5)
		mc = 1. / mc * 9

		if self.if_bias:
			x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
		m = self.pool(m)

		return x, m

class SAConvBlock(nn.Module):

	def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, dilation=1, bias=True):
		super(SAConvBlock, self).__init__()
		self.sparse_conv = SAConv(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, input):
		x, m = input
		x, m = self.sparse_conv((x, m))
		assert (m.size(1)==1)
		x = self.relu(x)
		
		return x, m


def make_blocks_from_names(names,in_dim,out_dim):
	layers = []
	if names[0] == "block1" or names[0] == "block2":
		layers += [SAConvBlock(in_dim, out_dim, 3,stride = 1)]
		layers += [SAConvBlock(out_dim, out_dim, 3,stride = 1)]
	else:
		layers += [SAConvBlock(in_dim, out_dim, 3,stride = 1)]
		layers += [SAConvBlock(out_dim, out_dim, 3,stride = 1)]
		layers += [SAConvBlock(out_dim, out_dim, 3,stride = 1)]
	return nn.Sequential(*layers)

#######
# Starting from here
#######

class hourglass(nn.Module):
	def __init__(self, inplanes):
		super(hourglass, self).__init__()

		self.conv1 = nn.Sequential(convbn(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1, dilation=1),
								nn.ReLU(inplace=True))
		self.conv2 = convbn(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1, dilation=1)

		self.conv3 = nn.Sequential(convbn(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1, dilation=1),
			nn.ReLU(inplace=True))

		self.conv4 = nn.Sequential(convbn(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1, dilation=1),
			nn.ReLU(inplace=True))

		self.conv5 = nn.Sequential(nn.ConvTranspose2d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
			nn.BatchNorm2d(inplanes*2)) 

		self.conv6 = nn.Sequential(nn.ConvTranspose2d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
			nn.BatchNorm2d(inplanes)) 

	def forward(self, x ,presqu, postsqu):
		
		out  = self.conv1(x) #in:1/4 out:1/8
		pre  = self.conv2(out) #in:1/8 out:1/8
		if postsqu is not None:
		   pre = F.relu(pre + postsqu, inplace=True)
		else:
		   pre = F.relu(pre, inplace=True)

		out  = self.conv3(pre) #in:1/8 out:1/16
		out  = self.conv4(out) #in:1/16 out:1/16

		# Each Houglass has its own structure. This is fine.
		if presqu is not None:
		   post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
		else:
		   post = F.relu(self.conv5(out)+pre, inplace=True) 

		out  = self.conv6(post)  #in:1/8 out:1/4

		return out, pre, post

class hourglass_l(nn.Module):
	def __init__(self, inplanes):
		super(hourglass_l, self).__init__()

		self.conv1 = nn.Sequential(convbn(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1, dilation=1),
								nn.ReLU(inplace=True))
		self.conv2 = convbn(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1, dilation=1)

		self.conv3 = nn.Sequential(convbn(inplanes*2, inplanes*4, kernel_size=3, stride=2, pad=1, dilation=1),
			nn.ReLU(inplace=True))

		self.conv4 = nn.Sequential(convbn(inplanes*4, inplanes*4, kernel_size=3, stride=1, pad=1, dilation=1),
			nn.ReLU(inplace=True))

		self.conv5 = nn.Sequential(nn.ConvTranspose2d(inplanes*4, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
			nn.BatchNorm2d(inplanes*2)) #+conv2

		self.conv6 = nn.Sequential(nn.ConvTranspose2d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
			nn.BatchNorm2d(inplanes)) #+x

	def forward(self, x ,presqu, postsqu):
		
		out  = self.conv1(x) #in:1/4 out:1/8
		pre  = self.conv2(out) #in:1/8 out:1/8
		if postsqu is not None:
		   pre = F.relu(pre + postsqu, inplace=True)
		else:
		   pre = F.relu(pre, inplace=True)

		out  = self.conv3(pre) #in:1/8 out:1/16
		out  = self.conv4(out) #in:1/16 out:1/16

		# Each Houglass has its own structure. This is fine.
		if presqu is not None:
		   post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
		else:
		   post = F.relu(self.conv5(out)+pre, inplace=True) 

		out  = self.conv6(post)  #in:1/8 out:1/4

		return out, pre, post

class SCADCNetGenerator(nn.Module):
	def __init__(self):
		super(SCADCNetGenerator, self).__init__()
		batchNorm_momentum = 0.1
		self.need_initialization = []

		self.sa1 = make_blocks_from_names(["block1"], 1,64)
		self.sa2 = make_blocks_from_names(["block2"], 64, 2)

		self.conv1 = nn.Sequential(convbn(1, 32, kernel_size=3, stride=1, pad=1, dilation=1), nn.ReLU(inplace=True))
		self.conv2 = nn.Sequential(convbn(32, 32, kernel_size=3, stride=1, pad=1, dilation=1), nn.ReLU(inplace=True))

		self.hg_0 = hourglass(32)
		self.hg_1 = hourglass(32)
		self.hg_2 = hourglass(32)

		self.classif1 = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
			nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
			nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1))
		self.classif2 = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
			nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
			nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1))
		self.classif3 = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
			nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1),
			nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1))
		self.count = 0


	def forward(self, in_disp, in_depth, sp_map, mask):

		fmap_1, m_1 = self.sa1((sp_map,mask))
		fmap_2, m_2 = self.sa2((fmap_1,m_1))
		
		## Using softmax for AMP
		AMP = F.softmax(fmap_2, dim=1)		
		AMP_dep = AMP[:,1,:,:].unsqueeze(1)
		in_1 = torch.sum((torch.cat((in_disp, in_depth), dim=1)*AMP), dim=1).unsqueeze(1)

		f_1 = self.conv1(in_1)
		f_2 = self.conv2(f_1) + f_1

		o_1, prev_1, post_1 = self.hg_0(f_2, None, None)
		o_1 = o_1 + f_2

		o_2, prev_2, post_2 = self.hg_1(o_1, prev_1, post_1)
		o_2 = o_2 + f_2

		o_3, prev_3, post_3 = self.hg_2(o_2, prev_1, post_2)
		o_3 = o_3 + f_2

		logits_1 = self.classif1(o_1) 	
		logits_2 = self.classif2(o_2) + logits_1 
		logits_3 = self.classif3(o_3) + logits_2 

		return logits_1, logits_2, logits_3, AMP_dep
#####
# Loss term definition
#####

class MaskedMSELoss(nn.Module):
	def __init__(self):
		super(MaskedMSELoss, self).__init__()

	def forward(self, pred, target):
		assert pred.dim() == target.dim(), "inconsistent dimensions"
		valid_mask = (target>0).detach()
		diff = target - pred
		diff = diff[valid_mask]
		self.loss = (diff ** 2).mean()
		return self.loss

if __name__ == '__main__':
	a = torch.ones((2,1,100,100))
	b = torch.ones((2,1,100,100))
	net = SCADCNetGenerator()
	out = net(a,b)