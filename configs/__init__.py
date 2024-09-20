import importlib
from argparse import Namespace
from ast import literal_eval
from util.net import get_timepc
from util.util import get_resources_occupation


def get_cfg(opt_terminal):
	"""
	* 解析命令行参数到cfg对象中,方便动态修改操作
	* 生成一个用于分布式训练的命令(cfg.command)
	* 处理通过命令行传入的自定义选项，每个选项的格式是 key=value
	* return : 返回一个包含命令参数、分布式训练的命令、自定义选项的对象
	"""
	# 配置文件路径，将路径中的文件扩展名（.py）去掉，然后将路径中的 / 替换为 .，使其变为模块名
	opt_terminal.cfg_path = opt_terminal.cfg_path.split('.')[0].replace('/', '.')
	
	# 使用 importlib.import_module() 动态导入这个模块 dataset_lib，从而加载配置文件内容。
	dataset_lib = importlib.import_module(opt_terminal.cfg_path)
	
	# dataset_lib.__dict__ 包含了该模块中的所有变量、函数和类。
	cfg_terms = dataset_lib.__dict__
	ks = list(cfg_terms.keys())
	for k in ks:
		if k.startswith('_'):
			# 删除私有变量
			del cfg_terms[k]

	# 将模块中的所有变量转化为一个 Namespace 对象 cfg，便于后续属性的动态修改
	cfg = Namespace(**dataset_lib.__dict__)

	# 用于将命令行选项的值覆盖或添加到配置中
	for key, val in opt_terminal.__dict__.items():
		cfg.__setattr__(key, val)

	if opt_terminal.__dict__.get('checkpoint_path', None) :
		cfg.model.model_kwargs.checkpoint_path = opt_terminal.__dict__.get('checkpoint_path',None)
	if opt_terminal.__dict__.get('model_name', None) :
		cfg.model.name = opt_terminal.__dict__.get('model_name',None)
	if opt_terminal.__dict__.get('batch_size', None) : 
		cfg.trainer.data.batch_size = opt_terminal.__dict__.get('batch_size', None)
	if opt_terminal.__dict__.get('data_path', None) : 
		cfg.data.root = opt_terminal.__dict__.get('data_path', None)

	# 使用字符串插值，生成一个用于分布式训练的命令
	cfg.command = f'python3 -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port --use_env run.py -c {cfg.cfg_path} -m {cfg.mode} --sleep {cfg.sleep} --memory {cfg.memory} --dist_url {cfg.dist_url} --logger_rank {cfg.logger_rank} {" ".join(cfg.opts)}'
	
	# 处理通过命令行传入的自定义选项，每个选项的格式是 key=value
	for opt in cfg.opts:
		# 不会发生拷贝，cfg_ghost 与 cfg 指向同一个对象
		cfg_ghost = cfg
		ks, v = opt.split('=')
		# 键可能是嵌套的，比如 model.layer1.units=64，需要在cfg中找到并设置layer.units = 64，且layer可能是字典或对象
		ks = ks.split('.')
		try:
			# 将v转换为适合的类型，转换失败则保持为字符串
			v = literal_eval(v)
		except:
			v = v
		for i, k in enumerate(ks) :
			if i == len(ks) - 1:
				if isinstance(cfg_ghost, dict):
					cfg_ghost[k] = v
				else:
					cfg_ghost.__setattr__(k, v)
			else:
				if k not in cfg_ghost:
					# 不在当前对象中，则新建一个对象
					cfg_ghost.__setattr__(k, Namespace())
				# 遍历下一层对象
				cfg_ghost = cfg_ghost.__dict__[k]
	
	# 任务开始时间
	cfg.task_start_time = get_timepc()
	# 任务开始时的资源占用
	cfg.task_start_allocation = get_resources_occupation()
	return cfg
