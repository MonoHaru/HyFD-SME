import os, sys
import argparse
from pprint import pprint


class Option:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ========================================
        #            General Options
        # ========================================
        self.parser.add_argument('--cuda_idx', type=str, default='cuda:0', help='cuda idx')
        self.parser.add_argument('--data_dir', type=str, default='./datasets', help='path to dataset')
        self.parser.add_argument('--ckpt', type=str, default='./checkpoints', help='path to save checkpoint')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether it is to evaluate the model')
        self.parser.add_argument('--model_name', type=str, default=None, 
                                 help='the model path will be evaluated')

        # ========================================
        #            Running Options
        # ========================================
        self.parser.add_argument('--random_seed', type=int, default=777)
        self.parser.add_argument('--epoch', type=int, default=500)
        self.parser.add_argument('--batch_size', type=int, default=256)
        self.parser.add_argument('--overlap_percentage', type=int, default=0, 
                                 help='overlapping percentage of data,. ex) 0, 10, 20, 30, 40, 50')
        self.parser.add_argument('--snr', default=None, 
                                 help='nosie snr of data,. ex) -4, -2, 0, 2, 4, None')
        self.parser.add_argument('--is_load', dest='is_load', action='store_true', help='whether to load exixsting model')
        # Learning Rate & Scheduler
        self.parser.add_argument('--T_0', type=int, default=100, help='first period of learning rate scheduler')
        self.parser.add_argument('--T_mult', type=int, default=1, help='coefficient period multiplication of learning rate scheduler')
        self.parser.add_argument('--eta_max', type=float, default=3e-3, help='last learning rate of scheduler')
        self.parser.add_argument('--T_up', type=int, default=10, help='linearly increasing of learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.9)

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self, makedir=True):
        self._initial()
        self.opt = self.parser.parse_args()
        self.opt.data_dir = os.path.join(self.opt.data_dir, '{}_percent_overlapping.csv'.format(self.opt.overlap_percentage))
        self.opt.exp = 'overlap_{}_per_{}_snr'.format(self.opt.overlap_percentage,
                                                      self.opt.snr)
        self.opt.snr = int(self.opt.snr)
        
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if makedir == True:
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
            self.opt.ckpt = ckpt
        
        self._print()
        
        return self.opt