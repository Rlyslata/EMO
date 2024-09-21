import os
import sys
import time
import logging
import shutil
import argparse
import torch
import psutil
import subprocess
from tensorboardX import SummaryWriter


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Unsupported value encountered.')



def run_pre(cfg):
	"""
	* from time : 程序启动倒计时
	* from memory : 循环检测机器中每个GPU占用是否过高, 占用过高则睡眠等待1s
	* return : none
	"""
	# from time
	if cfg.sleep > -1:
		for i in range(cfg.sleep):
			time.sleep(1)
			print('\rCount down : {} s'.format(cfg.sleep - 1 - i), end='')
	# from memory
	elif cfg.memory > -1:
		s_times = 0
		while True:
			os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Used > tmp')
			memory_used = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
			if memory_used[0] < 3000:
				os.system('rm tmp')
				break
			else:
				s_times += 1
				time.sleep(1)
				print('\rWaiting for {} s'.format(s_times), end='')


def makedirs(dirs, exist_ok=False):
	if not isinstance(dirs, list):
		dirs = [dirs]
	for dir in dirs:
		os.makedirs(dir, exist_ok=exist_ok)
	
	
def init_checkpoint(cfg):
	"""
	* cfg.trainer.checkpoint 模型保存目录 runs_emo
	* cfg.trainer.resume_dir==True 则从断点加载state_dict
	* 单机训练：保存日志到runs_emo/logdir = '{}_{}_{}_{}'.format(cfg.trainer.name, cfg.model.name, cfg.data.name, time.strftime("%Y%m%d-%H%M%S"))
	* 设置日志记录器cfg.logger， 和 SummaryWriter， 但分布式训练没有cfg.logger
	"""
	def rm_zero_size_file(path):
			files = os.listdir(path)
			for file in files:
				path = '{}/{}'.format(cfg.logdir, file)
				size = os.path.getsize(path)  # unit:B
				if os.path.isfile(path) and size < 8:
					os.remove(path)

	os.makedirs(cfg.trainer.checkpoint, exist_ok=True)

	# 从 checkpoint 加载
	if cfg.trainer.resume_dir:
		cfg.logdir = '{}/{}'.format(cfg.trainer.checkpoint, cfg.trainer.resume_dir)
		checkpoint_path = cfg.model.model_kwargs['checkpoint_path']
		if checkpoint_path == '':
			cfg.model.model_kwargs['checkpoint_path'] = '{}/latest_ckpt.pth'.format(cfg.logdir)
		else:
			cfg.model.model_kwargs['checkpoint_path'] = '{}/{}'.format(cfg.logdir, checkpoint_path.split('/')[-1])
		state_dict = torch.load(cfg.model.model_kwargs['checkpoint_path'], map_location='cpu')
		cfg.trainer.iter, cfg.trainer.epoch = state_dict['iter'], state_dict['epoch']
		cfg.trainer.topk_recorder = state_dict['topk_recorder']
	else:
		if cfg.master:
			# 单机训练
			logdir = '{}_{}_{}_{}'.format(cfg.trainer.name, cfg.model.name, cfg.data.name, time.strftime("%Y%m%d-%H%M%S"))
			cfg.logdir = '{}/{}'.format(cfg.trainer.checkpoint, logdir)
			os.makedirs(cfg.logdir, exist_ok=True)
			shutil.copy('{}.py'.format('/'.join(cfg.cfg_path.split('.'))), '{}/{}.py'.format(cfg.logdir, cfg.cfg_path.split('.')[-1]))
		else:
			cfg.logdir = None
		cfg.trainer.iter, cfg.trainer.epoch = 0, 0
		cfg.trainer.topk_recorder = dict(net_top1=[], net_top5=[], net_E_top1=[], net_E_top5=[])
	
	# 设置日志记录器，同时输出在stdout 与 文件中
	cfg.logger = get_logger(cfg) if cfg.master else None

	# tensorbord 用于记录数据和日志， comment会被追加在文件名末尾，以区分多次训练的日志
	cfg.writer = SummaryWriter(log_dir=cfg.logdir, comment='') if cfg.master else None
	log_msg(cfg.logger, f'==> Logging on master GPU: {cfg.logger_rank}')
	# rm_zero_size_file(cfg.logdir) if cfg.master else None


def log_cfg(cfg):
	
	def _parse_Namespace(cfg, base_str=''):
		ret = {}
		if hasattr(cfg, '__dict__'):
			for key, val in cfg.__dict__.items():
				if not key.startswith('_'):
					ret.update(_parse_Namespace(val, '{}.{}'.format(base_str, key).lstrip('.')))
		else:
			ret.update({base_str:cfg})
		return ret
	
	cfg_dict = _parse_Namespace(cfg)
	key_max_length = max(list(map(len, cfg_dict.keys())))
	excludes = ['writer.', 'logger.handlers']
	exclude_keys = []
	for k, v in cfg_dict.items():
		for exclude in excludes:
			if k.find(exclude) != -1:
				exclude_keys.append(k) if k not in exclude_keys else None
	# cfg_str = '\n'.join(
	# 	[(('{' + ':<{}'.format(key_max_length) + '} : {' + ':<{}'.format(key_max_length)) + '}').format(k, str(v)) for
	# 	 k, v in cfg_dict.items()])
	cfg_str = ''
	for k, v in cfg_dict.items():
		if k in exclude_keys:
			continue
		cfg_str += ('{' + ':<{}'.format(key_max_length) + '} : {' + ':<{}'.format(key_max_length) + '}').format(k, str(v))
		cfg_str += '\n'
	cfg_str = cfg_str.strip()
	cfg.cfg_dict, cfg.cfg_str = cfg_dict, cfg_str
	log_msg(cfg.logger, f'==> ********** cfg ********** \n{cfg.cfg_str}')


def get_logger(cfg, mode='a+'):
	"""
	return : 返回一个日志记录器， 输出在stdout 并记录在文件中， 文件记录位置为 '{}/log_{}.txt'.format(cfg.logdir, cfg.mode)
	"""
	log_format = '%(asctime)s - %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
	fh = logging.FileHandler('{}/log_{}.txt'.format(cfg.logdir, cfg.mode), mode=mode)
	fh.setFormatter(logging.Formatter(log_format))
	logger = logging.getLogger()
	logger.addHandler(fh)
	cfg.logger = logger
	return logger


def start_show(logger):
	logger.info('********************************************************************************')
	logger.info('==>   ======= ==    ==       ==     ============               ==            <==')
	logger.info('==>        == ==  ==           ==        ==           ====================   <==')
	logger.info('==>   ======= ===            ==  ==      ==           ==     ======     ==   <==')
	logger.info('==>   ==   ===========         ==        ==                    ==            <==')
	logger.info('==>   ======= ==                 ==      ==                    ==            <==')
	logger.info('==>     == == ==  ==           ==        ==                 == ==            <==')
	logger.info('==>        == ==    ==       ==     ============               ==            <==')
	logger.info('********************************************************************************')
	
	logger.info('********************************************************************************')
	logger.info('==>  =       =  =========  ========  =     =      =      =       =   ======  <==')
	logger.info('==>   =     =       =           ==   =     =     = =     = =     =  =    ==  <==')
	logger.info('==>    =   =        =         ==     =======    =====    =   =   =  =    ==  <==')
	logger.info('==>     = =         =       ==       =     =   =     =   =     = =  =        <==')
	logger.info('==>      =          =      ========  =     =  =       =  =       =   ======  <==')
	logger.info('********************************************************************************')


def able(ret, mark=False, default=None):
	return ret if mark else default


def log_msg(logger, msg, level='info'):
	if logger is not None:
		if msg is not None and level == 'info':
			logger.info(msg)


class AvgMeter(object):
	def __init__(self, name, fmt=':f', show_name='val', add_name=''):
		self.name = name
		self.fmt = fmt
		self.show_name = show_name
		self.add_name = add_name
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '[{name} {' + self.show_name + self.fmt + '}'
		fmtstr += (' ({' + self.add_name + self.fmt + '})]' if self.add_name else ']')
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, meters, default_prefix=""):
		self.iter_fmtstr_iter = '{}: {:>3.2f}% [{}/{}]'
		self.iter_fmtstr_batch = ' [{:<.1f}/{:<3.1f}]'
		self.meters = meters
		self.default_prefix = default_prefix

	def get_msg(self, iter, iter_full, epoch=None, epoch_full=None, prefix=None):
		entries = [self.iter_fmtstr_iter.format(prefix if prefix else self.default_prefix, iter / iter_full * 100, iter, iter_full, epoch, epoch_full)]
		if epoch:
			entries += [self.iter_fmtstr_batch.format(epoch, epoch_full)]
		for meter in self.meters.values():
			entries.append(str(meter)) if meter.count > 0 else None
		return ' '.join(entries)


def get_log_terms(log_terms, default_prefix=''):
	terms = {}
	for t in log_terms:
		t = {k: v for k, v in t.items()}
		t_name = t['name']
		terms[t_name] = AvgMeter(**t)
	progress = ProgressMeter(terms, default_prefix=default_prefix)
	return terms, progress


def update_log_term(term, val, n, master):
	term.update(val, n) if term and master else None


def accuracy(output, target, topk=(1,)) -> dict:
	maxk = max(topk)
	batch_size = target.size(0)
	# 类别数
	class_num = output.shape[1]
	assert class_num >= maxk, '最大类别数需 <= {class_num}'
	# input: 需要查找 top-k 元素的张量,
	# k: 指定要返回的最大或最小元素的数量,
	# dim (可选): 指定沿哪个维度查找 top-k 值。如果不指定，默认查找的是最后一个维度。
	# largest (可选，默认 True): 如果设置为 True，返回最大的前 k 个值；如果设置为 False，返回最小的前 k 个值。
	# sorted (可选，默认 True): 是否对返回的前 k 个值进行排序。如果 True，返回的值将按降序排序。
	# out (可选): 可以传入两个张量的元组，分别用来接收结果

	# @return: values: 包含前 k 个最大（或最小）值的张量, indices: 这些 top-k 值在输入张量中对应的索引。
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	pred_max_index = output.argmax(dim = 1).reshape(1,-1)
	correct = pred.eq(target.reshape(1, -1).expand_as(pred))

	
	# topk 预测准确数
	topk_num = [correct[:k].reshape(-1).float().sum(0) for k in topk]
	# topk 预测准确率
	topk_acc = [num * 100. / batch_size for num in topk_num]

	# 每个类别分别求 tp,tn,fp,fn
	tp,tn,fp,fn = [torch.zeros(class_num) for _ in range(4)]
	
	for cls in range(class_num) :
		pred_positive = (pred_max_index == cls)
		actual_positive = (target == cls)
		# 计算TP, TN, FP, FN
		tp[cls] = torch.sum(pred_positive & actual_positive).item()  # True Positive
		tn[cls] = torch.sum(~pred_positive & ~actual_positive).item() # True Negative
		fp[cls] = torch.sum(pred_positive & ~actual_positive).item()  # False Positive
		fn[cls] = torch.sum(~pred_positive & actual_positive).item()  # False Negative
		

	return  {
		# topk 预测准确率
		'topk_acc' : topk_num,
		# topk 预测正确的样本数
		'topk_num' : topk_acc,
		'tp' : tp,
		'tn' : tn,
		'fp' : fp,
		'fn' : fn,
		# 样本全体
		'top_all' : batch_size
	}


def get_cpu_and_memory_usage():
    # 获取当前进程的 PID
    pid = os.getpid()
    process = psutil.Process(pid)

    # 获取 CPU 占用率（百分比）
    cpu_usage = psutil.cpu_percent(interval=1)

    # 获取内存使用情况
    memory_info = process.memory_info()

	
    # print(f"CPU 使用率: {cpu_usage}%")
    # print(f"内存使用情况: {memory_info.rss / (1024 ** 2)} MB") 
	# rss 是常驻内存大小

    return {'cpu':cpu_usage, 'memory':memory_info.rss / (1024 ** 2)}

def get_gpu_memory_usage():
	"""
	使用 torch.cuda 获取已使用显存MB
	"""

	# print(f"已使用显存: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
	# print(f"最大显存占用: {torch.cuda.max_memory_allocated() / (1024 ** 2)} MB")

	return torch.cuda.memory_allocated() / (1024 ** 2)
 
def get_gpu_memory_nvidia_smi():
	"""
	使用 nvidia-smi 获取已使用显存MB
	"""
	result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
        encoding='utf-8')
	gpu_memory = int(result.strip())
	# print(f"nvidia-smi 显存使用: {gpu_memory} MB")
	return {'gpu': gpu_memory}

def get_resources_occupation() :
	res = get_cpu_and_memory_usage()
	res['gpu'] = get_gpu_memory_nvidia_smi()['gpu']
	return res

if __name__ == '__main__':
	# allocation = get_resources_occupation()
	# print(allocation)
	
	# meter = AvgMeter('name', fmt=':f', show_name='sum', add_name='count')
	# progress = ProgressMeter({'name':meter}, default_prefix='Test')

	# meter.update(1,1)
	# meter.update(1.1,1)
	# print(progress.get_msg(100,3000,1,30))
	# print(str(meter))
	output = torch.randn((5,2))
	target = torch.randint(0,2,(5,))
	res = accuracy(output=output, target=target,topk=(1,2))
	print(res)
