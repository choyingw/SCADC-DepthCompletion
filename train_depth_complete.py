#!/usr/bin/env python
# Author: Cho-Ying Wu, USC, March 2021
# Scene Completeness-Aware Lidar Depth Completion for Driving Scenario
# ICASSP 2021

import time
from options.options import AdvanceOptions
from models import create_model
from util.visualizer import Visualizer
from dataloaders.kitti_dataloader import KITTIDataset
import numpy as np
import random
import torch
import cv2

if __name__ == '__main__':
	train_opt = AdvanceOptions().parse(True)

	if not train_opt.test_path or not train_opt.train_path:
		raise ValueError('Please specify paths for both the training and testing data.')

	train_dataset = KITTIDataset(train_opt.train_path, type='train',
                modality='d2sm')
	test_dataset = KITTIDataset(train_opt.test_path, type='val',
            modality='d2sm')
	train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_opt.batch_size, shuffle=True,
            num_workers=train_opt.num_threads, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(train_opt.seed + work_id))

	test_opt = AdvanceOptions().parse(True)
	test_opt.phase = 'val'
	test_opt.batch_size = 1
	test_opt.num_threads = 1
	test_opt.serial_batches = True
	test_opt.no_flip = True
	test_opt.display_id = -1
	
	test_data_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=test_opt.batch_size, shuffle=True, num_workers=test_opt.num_threads, pin_memory=True)

	train_dataset_size = len(train_data_loader)
	print('#training images = %d' % train_dataset_size)
	test_dataset_size = len(test_data_loader)
	print('#test images = %d' % test_dataset_size)

	model = create_model(train_opt, train_dataset)
	model.setup(train_opt)
	visualizer = Visualizer(train_opt) # logger instance
	total_steps = 0
	for epoch in range(train_opt.epoch_count, train_opt.niter + 1):
		model.train()
		epoch_start_time = time.time()
		iter_data_time = time.time()
		epoch_iter = 0
		model.init_eval()
		iterator = iter(train_data_loader)
		while True:

			try: 
				next_batch = next(iterator)
			except StopIteration:
				break
			data, target = next_batch[0], next_batch[1]

			iter_start_time = time.time()
			if total_steps % train_opt.print_freq == 0:
				t_data = iter_start_time - iter_data_time
			total_steps += train_opt.batch_size
			epoch_iter += train_opt.batch_size
			model.set_new_input(data,target)
			model.optimize_parameters()

			if total_steps % train_opt.print_freq == 0:
				losses = model.get_current_losses()
				t = (time.time() - iter_start_time) / train_opt.batch_size
				visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

				message = model.print_depth_evaluation()
				visualizer.print_current_depth_evaluation(message)
				print()

			iter_data_time = time.time()

		print('End of epoch %d / %d \t Time Taken: %d sec' %   (epoch, train_opt.niter, time.time() - epoch_start_time))
		model.update_learning_rate()
		if epoch  and epoch % train_opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)

			model.eval()
			test_loss_iter = []
			epoch_iter = 0
			model.init_test_eval()
			with torch.no_grad():
				iterator = iter(test_data_loader)
				while True:
					try:
						next_batch = next(iterator)
					except IndexError:
						print("Corrupted data are catched! Discard this batch!")
						continue
					except StopIteration:
						break

					data, target = next_batch[0], next_batch[1]

					model.set_new_input(data,target)
					model.forward()
					model.test_depth_evaluation(test_opt)
					model.get_loss()
					epoch_iter += test_opt.batch_size
					losses = model.get_current_losses()
					print('test epoch {0:}, iters: {1:}/{2:} '.format(epoch, epoch_iter, len(test_dataset) * test_opt.batch_size), end='\r')
					message = model.print_test_depth_evaluation()
					visualizer.print_current_depth_evaluation(message) # print the loss, and error message to the log file
					print( # print on screen for fast validation
                  'RMSE= Curr: {result.rmse:.4f}(Avg: {average.rmse:.4f}) '
				  'MSE= Curr:{result.mse:.4f}(Avg: {average.mse:.4f}) '
				  'MAE= Curr:{result.mae:.4f}(Avg: {average.mae:.4f}) '
				  'Delta1= Curr:{result.delta1:.4f}(Avg: {average.delta1:.4f}) '
				  'Delta2= Curr:{result.delta2:.4f}(Avg: {average.delta2:.4f}) '
				  'Delta3= Curr:{result.delta3:.4f}(Avg: {average.delta3:.4f}) '
				  'REL= Curr:{result.absrel:.4f}(Avg: {average.absrel:.4f}) '
		  		  'Lg10= Curr:{result.lg10:.4f}(Avg: {average.lg10:.4f}) '.format(
                 result=model.test_result, average=model.test_average.average()))