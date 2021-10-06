# coding=utf-8
"""
Implementation of the paper "SplitNet: Divide and Co-training".
Author: Shuai Zhao
Contact: zhaoshuaimcc@gmail.com
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .mixup import mixup_data, mixup_criterion
# from . import pyramidnet, densenet, densenet3, resnexst
# from . import resnet, resnest, senet, efficientnet, shake_shake
# from . import efficientnet_pytorch as lukemelas_eff

__all__ = ['SplitNet']

from cifar import wide_resnet28_10


def _get_net(arch='resnet50'):
	"""get the backnbone net"""
	networks_obj_dict = {
		# For ImageNet
		# 'resnet50': resnet.resnet50,
		# 'resnet101': resnet.resnet101,
		# 'resnet152': resnet.resnet152,
		# 'resnet200': resnet.resnet200,
		# 'wide_resnet50_2': resnet.wide_resnet50_2,
		# 'wide_resnet50_3': resnet.wide_resnet50_3,
		# 'wide_resnet101_2': resnet.wide_resnet101_2,
		# 'resnext50_32x4d': resnet.resnext50_32x4d,
		# 'resnext101_32x4d': resnet.resnext101_32x4d,
		# 'resnext101_64x4d': resnet.resnext101_64x4d,
		# 'senet154': senet.senet154,
		# 'se_resnet50': senet.se_resnet50,
		# 'se_resnet50_B': senet.se_resnet50_B,
		# 'se_resnet101': senet.se_resnet101,
		# 'se_resnet152': senet.se_resnet152,
		# 'se_resnext101_64x4d': senet.se_resnext101_64x4d,
		# 'se_resnext101_64x4d_B': senet.se_resnext101_64x4d_B,
		# 'senet113': senet.senet113,
		# 'resnest101': resnest.resnest101,
		# 'resnest200': resnest.resnest200,
		# 'resnest269': resnest.resnest269,
		# 'resnexst50_4x16d': resnexst.resnexst50_4x16d,
		# 'resnexst50_8x16d': resnexst.resnexst50_8x16d,
		# 'resnexst50_4x32d': resnexst.resnexst50_4x32d,
		# 'resnexst101_8x32d': resnexst.resnexst101_8x32d,
		# For CIFAR/SVHN
		# 'resnet110': resnet.resnet110,
		# 'resnet164': resnet.resnet164,
		# 'resnext29_8x64d': resnet.resnext29_8x64d,
		# 'resnext29_16x64d': resnet.resnext29_16x64d,
		# 'wide_resnet16_8': resnet.wide_resnet16_8,
		# 'wide_resnet16_12': resnet.wide_resnet16_12,
		'wide_resnet28_10': wide_resnet28_10,
		# 'wide_resnet40_10': resnet.wide_resnet40_10,
		# 'wide_resnet52_8': resnet.wide_resnet52_8,
		# 'se_resnet110': senet.se_resnet110,
		# 'se_resnet164': senet.se_resnet164,
		# 'shake_resnet26_2x96d': shake_shake.shake_resnet26_2x96d,
		# 'pyramidnet164': pyramidnet.pyramidnet164,
		# 'pyramidnet272': pyramidnet.pyramidnet272,
	}
	assert arch in networks_obj_dict.keys()
	return networks_obj_dict[arch]


class SplitNet(nn.Module):
	def __init__(self,
					args,
					norm_layer=None,
					criterion=None,
					progress=True):
		super(SplitNet, self).__init__()

		self.split_factor = args.split_factor
		self.arch ='wide_resnet28_10'#args.arch
		self.loop_factor = 1#args.loop_factor
		self.is_train_sep = args.is_train_sep
		self.epochs = args.epochs

		# create models
		models = []
		if self.arch in ['resnet50', 'resnet101', 'resnet152', 'resnet200',
							'resnext50_32x4d', 'resnext101_32x4d',
							'resnext101_64x4d',
							'resnext29_8x64d', 'resnext29_16x64d',
							'resnet110', 'resnet164',
							'wide_resnet16_8', 'wide_resnet16_12',
							'wide_resnet28_10', 'wide_resnet40_10', 'wide_resnet52_8',
							'wide_resnet50_2', 'wide_resnet50_3', 'wide_resnet101_2']:

			model_kwargs = {'num_classes': args.num_classes,
							'norm_layer': norm_layer,
							'dataset': args.dataset,
							'split_factor': self.split_factor,
							'output_stride': args.output_stride,
							}

			for i in range(self.loop_factor):
				models.append(_get_net(self.arch)(pretrained=args.pretrained, **model_kwargs))

		else:
			raise NotImplementedError

		# Holds submodules in a list. Docs: https://pytorch.org/docs/stable/nn.html#modulelist
		self.models = nn.ModuleList(models)
		self.criterion = criterion
		if args.is_identical_init:
			self._identical_init()
		# add loss of the ensembled output
		self.is_ensembled_loss = False

		# using cot_loss (co-training loss)
		self.is_cot_loss = False

		self.num_classes = args.num_classes

	def forward(self, x, target=None, mode='train', epoch=0, streams=None):
		"""
		Output:
			ensemble_output: 	a tensor of shape [batch_size, num_classes]
			outputs: 			a tensor of shape [split_factor, batch_size, num_classes]
			ce_losses: 			a tensor of shape [split_factor, ]
			cot_loss: 			a tensor of shape [split_factor, ]
		"""
		outputs = []
		ce_losses = []

		if 'train' in mode:

			# Only do mixup for one time, otherwise it will violate the co-training loss and ensembled loss
			# if self.is_mixup:
			# 	x, y_a, y_b, lam = mixup_data(x, target, alpha=self.mix_alpha)

			# if self.is_diff_data_train:
			# 	# split the tensor of size [N, split_factor * C, H, W] into several [N, C, H, W] tensors
			# 	all_x = torch.chunk(x, chunks=self.loop_factor, dim=1)

			# How to run loop in parallel?
			# https://discuss.pytorch.org/t/using-streams-doesnt-seem-to-improve-performance/28575
			# https://discuss.pytorch.org/t/parallelize-simple-for-loop-for-single-gpu/33701/5
			# https://stackoverflow.com/questions/52498690/how-to-use-cuda-stream-in-pytorch
			# https://pytorch.org/docs/master/notes/multiprocessing.html
			if self.loop_factor == 1 or streams is None:
				for i in range(self.loop_factor):
					x_tmp = x
					output = self.models[i](x_tmp)
					# calculate the loss
					# print("output.size", output.shape)
					# print("target.size", target.shape)
					# print("output", output)
					# print("target", target)
					loss_now = self.criterion(output, target)
					ce_losses.append(loss_now)
					outputs.append(output)

			else:
				raise NotImplementedError

		elif mode in ['val', 'test']:
			for i in range(self.loop_factor):
				output = self.models[i](x)
				# if self.criterion is not None:
				# 	loss_now = self.criterion(output, target)
				# else:
				# 	loss_now = torch.zeros(1)
				loss_now = torch.zeros(1)
				outputs.append(output)
				ce_losses.append(loss_now)

		else:
			# for model summary
			for i in range(self.loop_factor):
				output = self.models[i](x)
			return torch.ones(1)

		# calculate the ensemble output
		ensemble_output = self._get_ensemble_output(outputs)
		ce_loss = torch.sum(torch.stack(ce_losses, dim=0))

		if 'val' in mode or 'test' in mode:
			return ensemble_output, torch.stack(outputs, dim=0), ce_loss

		# js divergence between all outputs
		if self.is_cot_loss:
			cot_loss = self._co_training_loss(outputs, self.cot_loss_choose, epoch=epoch)
		else:
			cot_loss = torch.zeros_like(ce_loss)

		return ensemble_output, torch.stack(outputs, dim=0), ce_loss, cot_loss

	def _get_ensemble_output(self, outputs):
		"""
		calculate the ensemble output.
		Currently, it only supports simple average or max.
		Args:
			outputs: a list of tensors, len(outputs) = split_factor.
		"""
		if False: #self.is_ensembled_after_softmax:
			if False:#self.is_max_ensemble:
				ensemble_output, _ = torch.max(F.softmax(torch.stack(outputs, dim=0), dim=-1), dim=0)
			else:
				ensemble_output = torch.mean(F.softmax(torch.stack(outputs, dim=0), dim=-1), dim=0)
			""" normalization, it doesn't work
			outputs_all = torch.stack(outputs, dim=0)
			outputs_std, outputs_mean = torch.std_mean(outputs_all, dim=1, keepdim=True)
			ensemble_output = torch.mean((outputs_all - outputs_mean) / outputs_std, dim=0)
			"""
		else:
			if False:#self.is_max_ensemble:
				ensemble_output, _ = torch.max(torch.stack(outputs, dim=0), dim=0)
			else:
				ensemble_output = torch.mean(torch.stack(outputs, dim=0), dim=0)

		return ensemble_output

	def _co_training_loss(self, outputs, loss_choose, epoch=0):
		"""calculate the co-training loss between outputs of different small networks
		"""
		weight_now = self.cot_weight
		if self.is_cot_weight_warm_up and epoch < self.cot_weight_warm_up_epochs:
			weight_now = max(self.cot_weight * epoch / self.cot_weight_warm_up_epochs, 0.005)

		if loss_choose == 'js_divergence':
			# the Jensen-Shannon divergence between p(x1), p(x2), p(x3)...
			# https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
			outputs_all = torch.stack(outputs, dim=0)
			p_all = F.softmax(outputs_all, dim=-1)
			p_mean = torch.mean(p_all, dim=0)
			H_mean = (- p_mean * torch.log(p_mean)).sum(-1).mean()
			H_sep = (- p_all * F.log_softmax(outputs_all, dim=-1)).sum(-1).mean()
			cot_loss = weight_now * (H_mean - H_sep)
		else:
			raise NotImplementedError

		return cot_loss

	def _identical_init(self):
		"""make the initial weights of networks the same"""
		with torch.no_grad():
			for i in range(1, self.split_factor):
				for (name1, m1), (name2, m2) in zip(self.models[i].named_parameters(),
														self.models[0].named_parameters()):
					if 'weight' in name1:
						m1.data.copy_(m2.data)
