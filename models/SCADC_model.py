import torch
from .base_model import BaseModel
from . import SCADC_networks
import numpy as np
import os
import math

class SCADCModel(BaseModel):
	def name(self):
		return 'SCADCNetModel'

	@staticmethod
	def modify_commandline_options(parser, is_train=True):

		# changing the default values
		if is_train:
			parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
		return parser

	def initialize(self, opt, dataset):
		BaseModel.initialize(self, opt)
		
		self.logits_1 = None
		self.logits_2 = None
		self.logits_3 = None

		self.loss_mse_1 = None
		self.loss_mse_2 = None
		self.loss_mse_3 = None
		
		# -----------------------
		self.result = None
		self.test_result = None
		self.average = None
		self.test_average = None
		self.mini_batch = opt.batch_size
		# --------------------

		self.isTrain = opt.isTrain
		# specify the training losses you want to print out. The program will call base_model.get_current_losses
		self.loss_names = ['mse_1','mse_2','mse_3','AMP','total']
		# specify the images you want to save/display. The program will call base_model.get_current_visuals
		self.visual_names = ['rgb_image','depth_image','mask','output']
		# specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
		self.model_names = ['SCADCNet']

		# load/define networks
		self.netSCADCNet = SCADC_networks.define_SCADCNet(init_type=opt.init_type,	init_gain= opt.init_gain, gpu_ids= self.gpu_ids)
        # define loss functions
		self.MSE = SCADC_networks.MaskedMSELoss()

		if self.isTrain:
			# initialize optimizers
			self.optimizers = []
			self.optimizer_SCADCNet = torch.optim.SGD(self.netSCADCNet.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
			self.optimizers.append(self.optimizer_SCADCNet)

	def set_new_input(self, input, target):
		self.raw_disp = input[:,0,:,:].to(self.device).unsqueeze(1)
		self.raw_depth = input[:,1,:,:].to(self.device).unsqueeze(1)
		self.gt_depth = target.to(self.device)
		self.sparse_depth = input[:,2,:,:].to(self.device).unsqueeze(1)
		self.sparse_mask = input[:,3,:,:].to(self.device).unsqueeze(1)
		self.dilate_mask = input[:,4,:,:].to(self.device).unsqueeze(1)

	def forward(self):
		self.logits_1,self.logits_2,self.logits_3, self.AMP_dep = self.netSCADCNet(self.raw_disp,self.raw_depth,self.sparse_depth,self.sparse_mask)
	
	def get_loss(self):
		self.loss_mse_1 = self.MSE(self.logits_1, self.gt_depth)
		self.loss_mse_2 = self.MSE(self.logits_2, self.gt_depth)
		self.loss_mse_3 = self.MSE(self.logits_3, self.gt_depth)
		self.loss_AMP = self.MSE(self.AMP_dep, self.dilate_mask)
		self.loss_total = self.loss_mse_1 + self.loss_mse_2 + self.loss_mse_3 +  self.loss_AMP

	def backward(self):
		self.loss_total.backward()

	def init_test_eval(self):
		self.test_result = Result()
		self.test_average = AverageMeter()

	def init_eval(self):
		self.result = Result()
		self.average = AverageMeter()
	
	def depth_evaluation(self):
		self.result.evaluate(self.logits_3, self.gt_depth)
		self.average.update(self.result, self.mini_batch)

	def test_depth_evaluation(self, opt):
		self.test_result.evaluate(self.logits_3, self.gt_depth)
		self.test_average.update(self.test_result, opt.batch_size)

	def init_disp_eval(self):
		self.disp_result = Result()
		self.disp_average = AverageMeter()
	def init_depth_eval(self):
		self.depth_result = Result()
		self.depth_average = AverageMeter()

	def test_in_disp_evaluation(self, opt):
		self.disp_result.evaluate(self.raw_disp, self.gt_depth)
		self.disp_average.update(self.disp_result, opt.batch_size)
	def test_in_depth_evaluation(self, opt):
		self.depth_result.evaluate(self.raw_depth, self.gt_depth)
		self.depth_average.update(self.depth_result, opt.batch_size)

	def print_depth_evaluation(self):
		message = 'RMSE={result.rmse:.4f}({average.rmse:.4f}) \
MAE={result.mae:.4f}({average.mae:.4f}) \
Delta1={result.delta1:.4f}({average.delta1:.4f}) \
REL={result.absrel:.4f}({average.absrel:.4f}) \
Lg10={result.lg10:.4f}({average.lg10:.4f})'.format(result=self.result, average=self.average.average())
		print(message)
		return message

	def print_test_depth_evaluation(self):
		message = 'RMSE={result.rmse:.4f}({average.rmse:.4f}) \
MAE={result.mae:.4f}({average.mae:.4f}) \
Delta1={result.delta1:.4f}({average.delta1:.4f}) \
Delta2={result.delta2:.4f}({average.delta2:.4f}) \
Delta3={result.delta3:.4f}({average.delta3:.4f}) \
REL={result.absrel:.4f}({average.absrel:.4f}) \
Lg10={result.lg10:.4f}({average.lg10:.4f})'.format(result=self.test_result, average=self.test_average.average())
		print(message)
		return message

	def optimize_parameters(self):
		self.forward()
		self.depth_evaluation()
		self.set_requires_grad(self.netSCADCNet, True)
		self.get_loss()
		self.optimizer_SCADCNet.zero_grad()
		self.backward()
		self.optimizer_SCADCNet.step()


####### Metrics ########
def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        valid_mask = target > 0

        output = output[valid_mask]
        target = target[valid_mask]

        new_output = output[target<=50]
        new_target = target[target<=50]  
        
        target = new_target
        output = new_output

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, n=1):
        self.count += n
        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, 
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        return avg
