#!/usr/bin/env python
# Author: Cho-Ying Wu, USC, March 2021
# Scene Completeness-Aware Lidar Depth Completion for Driving Scenario
# ICASSP 2021

import time
from options.options import AdvanceOptions
from models import create_model
from dataloaders.kitti_dataloader import KITTIDataset
import numpy as np
import random
import torch
import cv2
import utils
import os

def colored_depthmap(depth, d_min=None, d_max=None): # Color depth map
   if d_min is None:
	   d_min = np.min(depth)
   if d_max is None:
	   d_max = np.max(depth)
   depth_relative = (depth - d_min) / (d_max - d_min)
   return 255 * plt.cm.viridis(depth_relative)[:,:,:3] # H, W, C

def merge_into_row_with_pred_visualize(depth_target, depth_est): # Merge illustration function
   depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
   depth_pred_cpu = np.squeeze(depth_est.cpu().numpy())

   d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
   d_max = max(np.max(depth_target_cpu), np.min(depth_pred_cpu))
   depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
   depth_pred_col = colored_depthmap(depth_target_cpu, d_min, d_max)
   img_merge = np.hstack([depth_target_col,depth_pred_col])
   return img_merge

if __name__ == '__main__':
	test_opt = AdvanceOptions().parse(False)

	if not test_opt.test_path:
		raise ValueError('Please specify path for the testing data.')
	test_dataset = KITTIDataset(test_opt.test_path, type='val',
            modality='d2sm')

	test_opt.phase = 'val'
	test_opt.batch_size = 1
	test_opt.num_threads = 1
	test_opt.serial_batches = True
	test_opt.no_flip = True
	test_opt.display_id = -1

	test_data_loader = torch.utils.data.DataLoader(test_dataset,
		batch_size=1, shuffle=False, num_workers=test_opt.num_threads, pin_memory=True)

	test_dataset_size = len(test_data_loader)
	print('#test images = %d' % test_dataset_size)

	model = create_model(test_opt, test_dataset)
	model.eval()
	model.setup(test_opt)
	model.init_test_eval()

	# Will save results undet these folders.
	vis_path = 'vis_TEST2'
	if not os.path.exists(f'{vis_path}/colored'):
		os.makedirs(f'{vis_path}/colored')
	if not os.path.exists(f'{vis_path}/numerics'):
		os.makedirs(f'{vis_path}/numerics')
	if not os.path.exists(f'{vis_path}/gt'):
		os.makedirs(f'{vis_path}/gt')

	with torch.no_grad():
		iterator = iter(test_data_loader)
		iteration = 0
		while True:
			
			try:
				next_batch = next(iterator)
			except StopIteration:
				break

			data, target = next_batch[0], next_batch[1]
			model.set_new_input(data,target)
			model.forward()
			model.test_depth_evaluation(test_opt)
			model.get_loss()

			depth_target = model.gt_depth
			depth_est = model.logits_3

			## saving each frame
			depth_est[depth_est<0] = 0
			img_colored = utils.colored_depthmap(np.squeeze(depth_est.cpu().numpy()))
			img_uint16 = np.squeeze(depth_est.cpu().numpy())
			dilate_mask = np.squeeze(model.dilate_mask.cpu().numpy())		
			gt_for_save = np.squeeze(model.gt_depth.cpu().numpy())
			
			filename = f'{vis_path}/colored/'+str(iteration)+'.png'
			utils.save_image(img_colored, filename)
			
			filename = f'{vis_path}/numerics/{0:05d}'.format(iteration)+'.png'
			utils.save_image_16(img_uint16, filename)
			
			filename = f'{vis_path}/gt/'+str(iteration)+'.png'
			utils.save_image_16(gt_for_save, filename)

			iteration += 1

			print(
		  'RMSE= Curr: {result.rmse:.4f}(Avg: {average.rmse:.4f}) '
		  'MSE= Curr:{result.mse:.4f}(Avg: {average.mse:.4f}) '
		  'MAE= Curr:{result.mae:.4f}(Avg: {average.mae:.4f}) '
		  'Delta1= Curr:{result.delta1:.4f}(Avg: {average.delta1:.4f}) '
		  'Delta2= Curr:{result.delta2:.4f}(Avg: {average.delta2:.4f}) '
		  'Delta3= Curr:{result.delta3:.4f}(Avg: {average.delta3:.4f}) '
		  'REL= Curr:{result.absrel:.4f}(Avg: {average.absrel:.4f}) '
		  'Lg10= Curr:{result.lg10:.4f}(Avg: {average.lg10:.4f}) '.format(
		 result=model.test_result, average=model.test_average.average()))
