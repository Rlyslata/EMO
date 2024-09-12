import argparse
from configs import get_cfg
from util.net import init_training
from util.util import run_pre, init_checkpoint
from trainer import get_trainer
import warnings
warnings.filterwarnings("ignore")


def main():
	"""
	* 如果系统环境变量未设置 WORLD_SIZE, RANK, LOCAL_RANK 那么dist=False, 即会使用单机训练
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--cfg_path', default='configs/debug.py')
	parser.add_argument('-m', '--mode', default='train', choices=['train', 'test', 'ft'])
	parser.add_argument('--sleep', type=int, default=-1)
	parser.add_argument('--memory', type=int, default=-1)
	parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
	parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
	parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER,)
	cfg_terminal = parser.parse_args()

	# 解析参数，加载config/cls_emo.py中的参数，合并 cfg_terminal 为 cfg
	cfg = get_cfg(cfg_terminal)
	
	# --sleep 倒计时， --memory 检查GPU显存占用
	run_pre(cfg)
	
	init_training(cfg)
	# print(cfg.trainer.data.num_workers)
	# print(cfg.trainer.data.num_workers_per_gpu)
	# print('world_size = {cfg.world_size} rank = {cfg.rank} local_rank = {cfg.local_rank} dist = {cfg.dist}')

	#恢复或设置日志记录器
	init_checkpoint(cfg)

	trainer = get_trainer(cfg)
	# trainer.run()


if __name__ == '__main__':
	main()
