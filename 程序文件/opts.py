import argparse, os, json
import torch
import torchvision as tv
from utils import transforms

def get_opts():
    opt = argparse.Namespace()
    
    opt.task_name = ''
    opt.exp_name = 'exp1FeatherNetCC'
    opt.fold = 12
    opt.data_root = 'D:\\lcy\\feathernet\\dataset\\'
    #opt.data_list = 'data\\train.txt'     #运行时，改变文件名（存储图像路径的文件1.txt）
    opt.val = False   #验证标识,为真时，进行验证（测试）（600个项目）；为假时，进行测试（验证）（300个项目）
    
    opt.out_root = 'data\\opts\\'
    opt.out_path = os.path.join(opt.out_root,opt.exp_name,'fold{fold_n}'.format(fold_n=opt.fold))
    #opt.pretraind_path = 'D:\\lcy\\feathernet\\CVGNet\\mlfa\\data\\opts\\exp1FeatherNetCC\\fold2\\checkpoints\\116.pth' 
    #加载模型的存储路径
    
    ### Dataloader options ###
    opt.nthreads = 10         #windows系统下线程只能为0
    opt.batch_size = 32    #运行时改大点，原来是128
    opt.ngpu = 2

    ### Learning ###
    opt.freeze_epoch = 0
    opt.optimizer_name = 'Adam'          #使用Adam优化器
    opt.weight_decay = 0
    opt.lr = 2e-5
    opt.lr_decay_lvl = 0.5
    opt.lr_decay_period = 50
    opt.lr_type = 'cosine_repeat_lr'         #学习率的类型为cosine_repeat_lr
    opt.num_epochs=150          #原来是50
    opt.resume = False    #测试时，改为最佳模型的参数文件，如'116.pth.tar'
    opt.debug = 0
    ### Other ###  
    opt.manual_seed = 704
    opt.log_batch_interval=10
    opt.log_checkpoint = 1
    opt.net_type = 'FeatherNet_siam'            #使用ConvergedNet.invernetDLAS_A作为网络模型
    opt.pretrained = False    #加载指定的预训练模型的参数
    opt.classifier_type = 'arc_margin_5e-2'       #使用ArcMarginProduct(512, output_size, m=0.05)作为分类器
    opt.loss_type= 'cce'                   #使用交叉熵损失作为损失函数
    opt.alpha_scheduler_type = None             #alpha_scheduler定义为None,返回None
    opt.nclasses = 2
    opt.fake_class_weight = 1
    opt.visdom_port = 8097
    
    opt.git_commit_sha = '3ab79d6c8ec9b280f5fbdd7a8a363a6191fd65ce' 
    opt.train_transform = tv.transforms.Compose([
            transforms.MergeItems(True, p=0.1),
            #transforms.LabelSmoothing(eps=0.1, p=0.2),
            transforms.CustomRandomRotation(30, resample=2),
            transforms.CustomResize((125,125)),          #
            tv.transforms.RandomApply([
                transforms.CustomCutout(1, 25, 75)],p=0.1),
            #transforms.CustomGaussianBlur(max_kernel_radius=3, p=0.2),
            transforms.CustomRandomResizedCrop(112, scale=(0.5, 1.0)),        #
            transforms.CustomRandomHorizontalFlip(),
            tv.transforms.RandomApply([
                transforms.CustomColorJitter(0.25,0.25,0.25,0.125)],p=0.2),
            transforms.CustomRandomGrayscale(p=0.1),
            transforms.CustomToTensor(),
            transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

    opt.test_transform = tv.transforms.Compose([
            transforms.CustomResize((125,125)),
            transforms.CustomRotate(0),
            transforms.CustomRandomHorizontalFlip(p=0),
            transforms.CustomCrop((112,112), crop_index=0),
            transforms.CustomToTensor(),
            transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    
    
    return opt


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath', type=str, default = 'data/opts/', help = 'Path to save options')
    conf = parser.parse_args()
    opts = get_opts()
    save_dir = os.path.join(conf.savepath, opts.exp_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir,opts.exp_name + '_' + 'fold{0}'.format(opts.fold) + '_' + opts.task_name+'.opt')
    torch.save(opts, filename)
    print('Options file was saved to '+filename)
