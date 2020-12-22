from config import *
import os, sys, platform
from pynvml import *
sys.path.append(ROOT+'model_scripts/')
sys.path.append(ROOT+'tools/')

from data_tools import ELEDataset
from clozeTest_model import ClozeTestModel
from baseline_model import BaseLineModel
from preprocess import Preprocessor, Sample

import argparse
import time, random, json
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import  DistributedSampler
import transformers
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW


def gen_dataloaders(args):
	# preprocess raw data if pt data not prepared
	if 'albert' in args.pretrained_model:
		model_name = 'albert'
	elif 'roberta' in args.pretrained_model:
		model_name = 'roberta'
	elif 'xlnet' in args.pretrained_model:
		model_name = 'xlnet'
	elif 'bert' in args.pretrained_model:
		model_name = 'bert'

	data_collections = ['train', 'dev', 'test']
	for each in data_collections:
		args.data_dir = ROOT+'/data/raw/{}'.format(each)
		args.out_path = ROOT+'/data/pt/{}-{}.pt'.format(each, model_name)

		if not os.path.exists(args.out_path):
			preprocessor = Preprocessor(args)
			samples = preprocessor.preprocess()
			preprocessor.save(samples)
	  
	# load data and return DataLoader
	# train, val, test dataset
	train_dataset = ELEDataset(ROOT+'/data/pt/{}-{}.pt'.format('train', model_name), bool(args.debug))
	nb_train_samples = int(len(train_dataset)*0.9)
	nb_val_samples = len(train_dataset)-nb_train_samples
	train_dataset, val_dataset = random_split(train_dataset, [nb_train_samples, nb_val_samples])
	test_dataset = ELEDataset(ROOT+'/data/pt/{}-{}.pt'.format('dev', model_name), bool(args.debug))
	if args.local_rank == 0:
		args.LOGGER.logging("debug mode:{} train samples:{}  val samples:{}  test samples:{}".format(bool(args.debug), len(train_dataset), len(val_dataset),len(test_dataset)))
	# train, val, test data loaders
	if args.ngpu>1 and dist.is_available():
		train_sampler = DistributedSampler(train_dataset)
		val_sampler = DistributedSampler(val_dataset)
		test_sampler = DistributedSampler(test_dataset)
	else:
		train_sampler = None
		val_sampler = None
		test_sampler= None
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.ncpu, sampler=train_sampler)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.ncpu, sampler=val_sampler)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.ncpu, sampler=test_sampler)
 
	return train_loader, val_loader, test_loader, nb_train_samples

class Logger(object):
	def __init__(self, log_path):
		self.log_path = log_path
	def logging(self, content, print_=True, log_=True):
		# logging info during running
		if print_:
			print(content)
		if log_:
			with open(self.log_path, 'a+') as f_log:
				f_log.write(content + '\n')
	def log_super_params(self, args, print_=True, log_=True):
		params_dict = {
			'pretrained model name':args.pretrained_model,
			'encoder lr':args.encoder_lr,
			'decoder lr':args.decoder_lr,
			'batch size':args.batch_size,
			'accumulate steps':args.accumulate_steps,
			'max epochs':args.max_epoch,
			'total steps':args.total_steps*args.accumulate_steps,
			'fp16':bool(args.fp16),
			'num cpu used':args.ncpu,
			'num gpu used':args.ngpu,
			'debug mode':bool(args.debug),
		}
		# print info
		for k, v in params_dict.items():
			content = "{}:{}".format(k, v)
			if print_:
				print(content)
			if log_:
				with open(self.log_path, 'a+') as f_log:
					f_log.write(content + '\n')
					
	def log_hardware_info(self, print_=True, log_=True):
		info_dict = {}
		info_dict['platform_info'] = platform.platform()
		info_dict['processor_info'] = platform.processor()
		info_dict['python_version'] = sys.version
		info_dict['torch_version'] = torch.__version__
		info_dict['cuda_version'] = torch.version.cuda
		info_dict['transofrmers_version'] = transformers.__version__
		# gpu info
		nvmlInit()
		deviceCount = nvmlDeviceGetCount()
		info_dict['gpu_info'] = [nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)) for i in range(deviceCount)]
		# print info
		for k,v in info_dict.items():
			content = "{}:{}".format(k, v)
			if print_:
				print(content)
			if log_:
				with open(self.log_path, 'a+') as f_log:
					f_log.write(content+'\n')

def train(args):
	epoch_loss = []
	step_loss_li = []
	step_acc_num=0
	step_que_num=0
	train_acc_num=0
	train_que_num=0
	args.MODEL.train()
	
	for data_batch in args.TRAIN_LOADER:
		args.current_step += 1
		with autocast(enabled=bool(args.fp16)):
			article, option, answer, article_mask, option_mask, mask, blank_pos, sample_name = data_batch
			article = article.to(args.DEVICE)
			option = option.to(args.DEVICE)
			answer = answer.to(args.DEVICE)
			article_mask = article_mask.to(args.DEVICE)
			option_mask = option_mask.to(args.DEVICE)
			mask = mask.to(args.DEVICE)
			blank_pos = blank_pos.to(args.DEVICE)
			data_batch = (article, option, answer, article_mask, option_mask, mask, blank_pos, sample_name)
			
			batch_size, option_num, out = args.MODEL(data_batch)
			
			target = answer.view(-1, )
			# calculate loss
			step_loss = args.LOSS_FUNC(out, target)
			step_loss = step_loss.view(batch_size, option_num) * mask
			# replace nan to 0
			step_loss = torch.where(torch.isnan(step_loss), torch.full_like(step_loss, 0), step_loss)
			step_loss = step_loss.sum() / (mask.sum() if not mask.sum() == 0 else 1)
			step_loss /= args.accumulate_steps
			args.GRADSCALER.scale(step_loss).backward()
			step_loss_li.append(step_loss.item())

			# accuracy number
			acc_num = (torch.argmax(out, -1) == target).float()
			acc_num = acc_num.view(batch_size, option_num) * mask
			acc_num = acc_num.sum(-1)
			step_acc_num += acc_num.sum().detach()
			step_que_num += mask.sum().detach()
			
			
			if args.current_step%args.accumulate_steps == 0:
				args.GRADSCALER.step(args.OPTIMIZER)
				args.GRADSCALER.update()
				args.SCHEDULER.step()
				args.OPTIMIZER.zero_grad()
				# logging
				step_loss = torch.sum(torch.Tensor(step_loss_li)).item()
				acc = step_acc_num/step_que_num*100
				if args.local_rank == 0:
					args.LOGGER.logging("epoch[{}] global step[{}] train loss:{:.5f} train acc:{:.4f}%".format(args.current_epoch,args.current_step, step_loss, acc))

				# zero loss and acc/que num
				epoch_loss.append(step_loss)
				step_loss_li = []
				train_acc_num += step_acc_num
				train_que_num += step_que_num
				step_acc_num=0
				step_que_num=0

	epoch_avg_loss = torch.mean(torch.Tensor(epoch_loss))
	acc = train_acc_num/train_que_num*100
	if args.local_rank == 0:
		args.LOGGER.logging("epoch[{}] average train loss:{:.5f} train acc:{:.4f}".format(args.current_epoch, epoch_avg_loss.item(), acc))
		args.LOGGER.logging("epoch[{}] training complete.".format(args.current_epoch))
	
	return epoch_loss

def validation_test(args, mode = 'val'):
	args.MODEL.eval()
	val_loss = []
	val_acc_num = 0
	val_que_num = 0
	dataloader = args.VAL_LOADER if mode=='val' else args.TEST_LOADER
	with torch.no_grad():
		for data_batch in dataloader:
			article, option, answer, article_mask, option_mask, mask, blank_pos, sample_name = data_batch
			article = article.to(args.DEVICE)
			option = option.to(args.DEVICE)
			answer = answer.to(args.DEVICE)
			article_mask = article_mask.to(args.DEVICE)
			option_mask = option_mask.to(args.DEVICE)
			mask = mask.to(args.DEVICE)
			blank_pos = blank_pos.to(args.DEVICE)
			data_batch = (article, option, answer, article_mask, option_mask, mask, blank_pos, sample_name)
			
			batch_size, option_num, out = args.MODEL(data_batch)
			
			target = answer.view(-1, )
			# calculate loss
			step_loss = args.LOSS_FUNC(out, target)
			step_loss = step_loss.view(batch_size, option_num) * mask
			# replace nan to 0
			step_loss = torch.where(torch.isnan(step_loss), torch.full_like(step_loss, 0), step_loss)
			step_loss = step_loss.sum() / (mask.sum() if not mask.sum() == 0 else 1)
			val_loss.append(step_loss.item())
			# accuracy number
			acc_num = (torch.argmax(out, -1) == target).float()
			acc_num = acc_num.view(batch_size, option_num) * mask
			acc_num = acc_num.sum(-1)
			val_acc_num += acc_num.sum().detach()
			val_que_num += mask.sum().detach()
	avg_loss = torch.mean(torch.Tensor(val_loss)).item()
	accuracy = val_acc_num/val_que_num*100
	if args.local_rank == 0:
		if mode == 'val':
			args.LOGGER.logging("epoch[{}] val_loss={:.5f} val_accuracy={:.4f}%".format(args.current_epoch, avg_loss, accuracy))
			args.LOGGER.logging("epoch[{}] validation complete.".format(args.current_epoch))
		else:
			args.LOGGER.logging("test_loss={:.5f} test_accuracy={:.4f}%".format(avg_loss, accuracy))
			args.LOGGER.logging("test complete.")
	return avg_loss, accuracy.item()
	
def main(args):
	# seed all
	torch.manual_seed(42)
	np.random.seed(42)
	random.seed(42)
	if args.ngpu > 0:
		torch.cuda.manual_seed_all(42)
	# logger
	log_path = args.log_ckpt_dir + 'log.txt'
	args.LOGGER = Logger(log_path)
	
	# set device
	args.LOGGER.logging("setting device...")
	if args.local_rank != -1 and dist.is_available():
		dist.init_process_group(backend='nccl')
		args.local_rank = dist.get_rank()
		torch.cuda.set_device(args.local_rank)
		args.DEVICE = torch.device('cuda', args.local_rank)
	else:
		args.DEVICE = torch.device('cuda')
		args.local_rank = 0
		args.ngpu = 1
	if args.batch_size % dist.get_world_size() != 0:
		raise ValueError("batch size must be divisible by the world size")
	
	# get data loaders
	args.LOGGER.logging("local rank[{}]:preprocessing data and getting data loaders...".format(args.local_rank))
	args.TRAIN_LOADER, args.VAL_LOADER, args.TEST_LOADER, nb_train_samples = gen_dataloaders(args)
	
	# model
	args.LOGGER.logging("local rank[{}]:preparing model and initialize weights...".format(args.local_rank))
	args.MODEL = BaseLineModel(args.pretrained_model, args.debug) if bool(args.use_baseline) else ClozeTestModel(args.pretrained_model, args.debug)
	args.MODEL.to(args.DEVICE)
	if args.ngpu > 1:
		args.MODEL = DDP(args.MODEL, device_ids=[args.local_rank], output_device=args.local_rank)
	# TODO: init weight(or not)
	
	# optimizer and scheduler
	args.LOGGER.logging("local rank[{}]:preparing optimizer and scheduler...".format(args.local_rank))
	
	embeddings_params_id = []
	encoder_params_id = []
	for name, param in args.MODEL.named_parameters():
		if 'embeddings' in name:
			embeddings_params_id.append(id(param))
		if 'encoder' in name:
			encoder_params_id.append(id(param))
	encoder_embeddings_params = filter(lambda x:id(x) in embeddings_params_id+encoder_params_id,args.MODEL.parameters())
	rest_params = filter(lambda x:id(x) not in embeddings_params_id+encoder_params_id,args.MODEL.parameters())
	params = [
		{'params':encoder_embeddings_params, 'lr':args.encoder_lr},
		{'params':rest_params, 'lr': args.decoder_lr},
	] # for diffirent parameters using diffirent learning rate

	args.OPTIMIZER = AdamW(params=params)
	args.total_steps= int(nb_train_samples/(args.batch_size*args.accumulate_steps)*args.max_epoch)
	args.SCHEDULER = get_linear_schedule_with_warmup(args.OPTIMIZER, num_warmup_steps=int(args.total_steps/20), num_training_steps=args.total_steps)
	
	# backward scaler
	args.GRADSCALER = GradScaler(enabled=bool(args.fp16))
	
	# loss function
	args.LOSS_FUNC = nn.CrossEntropyLoss(reduction='none')
	
	# train and eval
	state = {'train_loss':[], 'val_loss':[], 'val_accuracy':[]}
	args.current_step = 0
	args.current_epoch = 0
	args.best_acc = 0
	args.patience = 2
	if bool(args.debug):
		args.max_epoch = 1
	
	# log hardware and super params info
	if args.local_rank == 0:
		args.LOGGER.log_hardware_info()
		args.LOGGER.log_super_params(args)
		
	for epoch in range(args.max_epoch):
		args.current_epoch += 1
		if args.local_rank == 0:
			args.LOGGER.logging("=====EPOCH{}/{}=====".format(args.current_epoch, args.max_epoch))
			args.LOGGER.logging("TRAINGING...")
		train_loss = train(args) # train
		if args.local_rank == 0:
			args.LOGGER.logging("VALIDATING...")
		val_loss, accuracy = validation_test(args, mode='val') # validation
		
		state['train_loss']+=train_loss
		state['val_loss'].append(val_loss)
		state['val_accuracy'].append(accuracy)
		
		# early stopping
		if args.best_acc < accuracy:
			args.best_acc = accuracy
		else:
			args.patience -= 1
		if args.patience == 0:
			if args.local_rank == 0:
				args.LOGGER.logging("EARLY STOPPING.")
			break
	if args.local_rank==0:
		args.LOGGER.logging("TRAIN AND VALIDATION COMPLETE.")

	# test
	if args.local_rank == 0:
		args.LOGGER.logging("TESTING...")
	_, accuracy = validation_test(args, mode='test')
	state['test_accuracy'] = accuracy

	# save state dict
	if args.local_rank == 0:
		args.LOGGER.logging("saving train state dict...")
		train_state_path = args.log_ckpt_dir + 'state.json'
		with open(train_state_path, 'w', encoding='utf-8') as f:
			json.dump(state, f)
			
	# save model
	if args.local_rank == 0:
		args.LOGGER.logging("saving model...")
		str_acc = str(state['test_accuracy']).replace(".", "")[:6]
		model_saved_path = args.log_ckpt_dir+"{}+{}+{}+{}+{}+{}+{}+{}.pt".format(args.pretrained_model, args.use_baseline, str(args.encoder_lr), str(args.decoder_lr), str(args.batch_size), str(args.accumulate_steps), str(args.current_epoch), str_acc)
		with open(model_saved_path, 'wb') as f:
			state_dict = args.MODEL.module.state_dict() if bool(args.fp16) else args.MODEL.state_dict()
			torch.save(state_dict, f)
		args.LOGGER.logging("all tasks complete.")
	sys.exit()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--task_name", type=str, default='ASC21ELE', help='task name of the training, do not change.')
	parser.add_argument("--pretrained_model", type=str, default='bert-large-uncased', help='choose between bert-large-uncased, albert-xlarge-v2 and roberta-large.')
	parser.add_argument("--encoder_lr",type=float, default=1e-5, help='learning rate for encoder')
	parser.add_argument("--decoder_lr", type=float, default=1e-4, help='learning rate for decoder, usually 10 times of the encoder_lr')
	parser.add_argument("--accumulate_steps", type=int, default=32, help='accumulation steps when training')
	parser.add_argument("--max_epoch", type=int, default=3, help='max epoch')
	parser.add_argument("--batch_size", type=int, default=2, help='batch size of data')
	parser.add_argument("--fp16", type=int, default=0, help='whether to use half precision, 1 to use and 0 not to')
	parser.add_argument("--local_rank", type=int, default=-1, help='local rank of device, do not change.')
	parser.add_argument("--debug", type=int, default=0, help='1 to set the debug mode to run a fast test if there is any problem in the program, 0 to disable debug.')
	parser.add_argument("--use_baseline", type=int, default=1, help='1 to use baseline model and 0 to use advanced model, in this version we only use baseline model.')
	args = parser.parse_args()
	args.ngpu = torch.cuda.device_count()
	args.ncpu = cpu_count()
	args.log_ckpt_dir = ROOT+'/experiments/{}+{}+{}+{}/'.format(args.pretrained_model,"use-baseline" if bool(args.use_baseline) else "not-use-baseline", "debug-mode" if bool(args.debug) else "not-debug-mode",time.strftime('%Y%m%d-%H%M%S'))
	print("gpu num:",args.ngpu)
	print("cpu num:",args.ncpu)
	os.makedirs(args.log_ckpt_dir, exist_ok=True)
	# main
	main(args)
	sys.exit()