"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
import torch

sys.path.append(".")
sys.path.append("..")

from options.train_options_twinnet import TrainOptions
from training.coach_twinnet import Coach 

def main():
	opts = TrainOptions().parse()
	create_initial_experiment_dir(opts)
	coach = Coach(opts)
	coach.train()
 
def create_initial_experiment_dir(opts):
	if not os.path.exists(opts.exp_dir):
		# raise Exception('{} already exists'.format(opts.exp_dir))
		os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)
 

if __name__ == '__main__':
	main()
