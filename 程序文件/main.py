import argparse,json,random,os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision as tv

from trainer import Model
from opts import get_opts
def main():
    
    
    opt = get_opts()
    
     
    # Fix seed
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)
    cudnn.benchmark = True           #可以大大提升卷积神经网络的运行速度
    
    # Create working directories
    try:
        os.makedirs(opt.out_path)     #os.makedirs() 方法用于递归创建目录
        os.makedirs(os.path.join(opt.out_path,'checkpoints'))
        os.makedirs(os.path.join(opt.out_path,'log_files'))
        print( 'Directory {} was successfully created.'.format(opt.out_path))
                   
    except OSError:
        print( 'Directory {} already exists.'.format(opt.out_path))
        pass
    
    
    # Training
    M = Model(opt)
    M.train()
    
    #M.test()
    
if __name__ == '__main__':
    main()


