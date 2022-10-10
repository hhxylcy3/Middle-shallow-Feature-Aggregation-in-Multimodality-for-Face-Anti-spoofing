import torch
import torch.nn as nn
import numpy as np
import time, os
import math
from utils.profile import count_params
from visdom import Visdom
from tools.benchmark import compute_speed, stat
import models, datasets, utils


#定义模型、分类器、学习率、优化器、载入数据、训练模型
class Model:
    def __init__(self, opt):
        self.opt = opt
        self.val=opt.val
        self.device = torch.device("cuda" if opt.ngpu else "cpu")
        #定义模型、分类器
        self.model, self.classifier = models.get_model(opt.net_type, 
                                                       opt.classifier_type, 
                                                       opt.pretrained,
                                                       int(opt.nclasses))
        
        
        self.model = self.model.to(self.device)    #FeatherNet_siam模型加载到GPU上运行 
        
        
        
        
        self.classifier = self.classifier.to(self.device)   #ArcMarginProduct分类器

        if opt.ngpu>1:
            self.model = nn.DataParallel(self.model)           
        
          
            
        #定义损失函数    
        self.loss = models.init_loss(opt.loss_type)    #CrossEntropyLoss损失函数
        self.loss = self.loss.to(self.device)
#定义优化器、调整学习率
        self.optimizer = utils.get_optimizer(self.model, self.opt)    #Adam优化器
        self.lr_scheduler = utils.get_lr_scheduler(self.opt, self.optimizer)   #'cosine_repeat_lr学习率衰减策略
        self.alpha_scheduler = utils.get_margin_alpha_scheduler(self.opt)    #参数为 None,返回None
        if opt.pretrained:   #加载指定的预训练模型的参数
            weights = torch.load('D:\\lcy\\feathernet\\CVGNet\\mlfa\\data\\opts\\exp1FeatherNetCC\\75_best.pth.tar', map_location='cpu')
            self.model.load_state_dict({k.replace('module.',''):v for k,v in weights['model_state_dict'].items()}) 
            self.classifier.load_state_dict({k.replace('module.',''):v for k,v in weights['classifier_state_dict'].items()}) 
            self.optimizer.load_state_dict({k.replace('module.',''):v for k,v in weights['optimizer'].items()}) 
            self.lr_scheduler.load_state_dict({k.replace('module.',''):v for k,v in weights['lr_scheduler'].items()})   
        
              
        
#载入数据
        self.train_loader = datasets.generate_loader(opt,'train') 
        self.test_loader = datasets.generate_loader(opt,'test') 
        self.val_loader = datasets.generate_loader(opt,'val')
        
        self.epoch = 0
        self.best_epoch = False
        self.training = False
        self.state = {}
        
#AverageMeter():功能---统计值、均值、总和、数量
        self.train_loss = utils.AverageMeter()
        self.test_loss  = utils.AverageMeter()
        self.batch_time = utils.AverageMeter()
        #if self.opt.loss_type in ['cce', 'bce', 'mse', 'arc_margin']:
            #self.test_metrics = utils.AverageMeter()
        #else:
            #self.test_metrics = utils.ROCMeter()   
        self.test_metrics = utils.ROCMeter()
        self.best_test_loss = utils.AverageMeter()                    
        self.best_test_loss.update(np.array([np.inf]))     #np.inf表示正无穷，没有确切的含义

        '''
        self.visdom_log_file = os.path.join(self.opt.out_path, 'log_files', 'visdom.log')
        self.vis = Visdom(port = opt.visdom_port,
                          log_to_filename=self.visdom_log_file,
                          env=opt.exp_name + '_' + str(opt.fold))

        self.vis_loss_opts = {'xlabel': 'epoch', 
                              'ylabel': 'loss', 
                              'title':'losses', 
                              'legend': ['train_loss', 'val_loss']}

        self.vis_tpr_opts = {'xlabel': 'epoch', 
                              'ylabel': 'tpr', 
                              'title':'val_tpr', 
                              'legend': ['tpr@fpr10-2', 'tpr@fpr10-3', 'tpr@fpr10-4']}

        self.vis_epochloss_opts = {'xlabel': 'epoch', 
                              'ylabel': 'loss', 
                              'title':'epoch_losses', 
                              'legend': ['train_loss', 'val_loss']}'''


    def train(self):
        
        # Init Log file
        if self.opt.resume:
            self.log_msg('resuming...\n')
            # Continue training from checkpoint
            self.load_checkpoint()
        else:
             self.log_msg()
        
        count_params(self.model )  #统计模型参数
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print('total_params',pytorch_total_params)
        


        for epoch in range(self.epoch, self.opt.num_epochs):   #重复训练n个迭代
            self.epoch = epoch                                             

            self.lr_scheduler.step()
            self.train_epoch()    #训练
            self.test_epoch()      #测试
            self.log_epoch()      #写入测试结果
            #self.vislog_epoch()
            self.create_state()       #创建保存的参数类型
            self.save_state()       #保存参数
     
    def test(self):
        self.load_checkpoint()
        self.test_epoch()
        self.log_epoch()
    
    
    def train_epoch(self):
        """
        Trains model for 1 epoch
        """
        self.model.train()
        self.classifier.train()
        self.training = True
        torch.set_grad_enabled(self.training)
        self.train_loss.reset()
        self.batch_time.reset()
        time_stamp = time.time()
        self.batch_idx = 0
        for batch_idx, (rgb_data, depth_data,ir_data, target) in enumerate(self.train_loader):
            
            self.batch_idx = batch_idx
            rgb_data = rgb_data.to(self.device)
            depth_data = depth_data.to(self.device)
            ir_data = ir_data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            
            output = self.model(rgb_data, depth_data, ir_data)
            if isinstance(self.classifier, nn.Linear):
                output = self.classifier(output)
            else:
                if self.alpha_scheduler:
                    alpha = self.alpha_scheduler.get_alpha(self.epoch)
                    output = self.classifier(output, target, alpha=alpha)
                else:
                    output = self.classifier(output, target)

            if self.opt.loss_type == 'bce':
                target = target.float()
                loss_tensor = self.loss(output.squeeze(), target)
            else:
                loss_tensor = self.loss(output, target)

            loss_tensor.backward()   

            self.optimizer.step()

            self.train_loss.update(loss_tensor.item())
            self.batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()
            
            self.log_batch(batch_idx)   #打印该批次的训练时间/损失和准确率
            #self.vislog_batch(batch_idx)
            
    def test_epoch(self):
        """
        Calculates loss and metrics for test set
        """
        self.training = False
        torch.set_grad_enabled(self.training)
        self.model.eval()
        self.classifier.eval()
        
        self.batch_time.reset()
        self.test_loss.reset()
        self.test_metrics.reset()
        time_stamp = time.time()
        if self.val:       #opt.val为真时，加载验证数据，为假时加载测试数据；
            test_data = self.val_loader
        else:
            test_data = self.test_loader
        
        
        for batch_idx, (rgb_data, depth_data,ir_data, target) in enumerate(test_data):
            rgb_data = rgb_data.to(self.device)
            depth_data = depth_data.to(self.device)
            ir_data = ir_data.to(self.device)
            target = target.to(self.device)

            output = self.model(rgb_data, depth_data,ir_data)
            output = self.classifier(output)  #学习全连接层的参数 ，使分类结果更接近于真实值
            if self.opt.loss_type == 'bce':
                target = target.float()
                loss_tensor = self.loss(output.squeeze(), target)
            else:
                loss_tensor = self.loss(output, target)
            self.test_loss.update(loss_tensor.item())
            #深度学习中通常所说的分类器值指的是softmaxloss前面的fc层，学习分类器说的就是学习fc层的参数，没有说学习loss的吧。分类器的作用是进行相似性度量，fc层中是WxF的形式，也就是内积形式的相似性度量，然后通过softmax计算最大后验概率。

            if self.opt.loss_type == 'cce' or self.opt.loss_type == 'focal_loss':
                output = torch.nn.functional.softmax(output, dim=1)   #softmax预测概率值输出
            elif self.opt.loss_type == 'bce':
                output = torch.sigmoid(output)

            self.test_metrics.update(target.cpu().numpy(), output.cpu().numpy())

            self.batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()
            
            self.log_batch(batch_idx)    #打印该批的训练时间、损失和准确率
            #self.vislog_batch(batch_idx)
            #self.vislog_batch(batch_idx)
            if self.opt.debug and (batch_idx==10):
                print('Debugging done!')
                break;

        self.best_epoch = self.test_loss.avg < self.best_test_loss.val
        if self.best_epoch:
             #self.best_test_loss.val is container for best loss, 
            # n is not used in the calculation
            self.best_test_loss.update(self.test_loss.avg, n=0)
     
    def calculate_metrics(self, output, target):   
        """
        Calculates test metrix for given batch and its input
        """
        t = target
        o = output
            
        if self.opt.loss_type == 'bce':
            accuracy = (t.byte()==(o>0.5)).float().mean(0).cpu().numpy()  
            batch_result.append(binary_accuracy)
        
        elif self.opt.loss_type == 'cce':
            top1_accuracy = (torch.argmax(o, 1)==t).float().mean().item()
            batch_result.append(top1_accuracy)
        else:
            raise Exception('This loss function is not implemented yet')
                
        return batch_result    
    
    def log_batch(self, batch_idx):
        if batch_idx % self.opt.log_batch_interval == 0:
            cur_len = len(self.train_loader) if self.training else len(self.test_loader)
            cur_loss = self.train_loss if self.training else self.test_loss
            
            output_string = 'Train ' if self.training else 'Test '
            output_string +='Epoch {}[{:.2f}%]: [{:.2f}({:.3f}) s]\t'.format(self.epoch,
                                                                          100.* batch_idx/cur_len, self.batch_time.val,self.batch_time.avg)
            
            loss_i_string = 'Loss: {:.5f}({:.5f})\t'.format(cur_loss.val, cur_loss.avg)
            output_string += loss_i_string
                    
            if not self.training:
                output_string+='\n'

                metrics_i_string = 'Accuracy: {:.5f}\t'.format(self.test_metrics.get_accuracy())
                output_string += metrics_i_string
                
            print(output_string)
    
    def vislog_batch(self, batch_idx):
        if batch_idx % self.opt.log_batch_interval == 0:
            loader_len = len(self.train_loader) if self.training else len(self.test_loader)
            cur_loss = self.train_loss if self.training else self.test_loss
            loss_type = 'train_loss' if self.training else 'val_loss'
            
            x_value = self.epoch + batch_idx / loader_len
            y_value = cur_loss.val
            self.vis.line([y_value], [x_value], 
                            name=loss_type, 
                            win='losses', 
                            update='append')
            self.vis.update_window_opts(win='losses', opts=self.vis_loss_opts)
    
    def log_msg(self, msg=''):
        mode = 'a' if msg else 'w'
        f = open(os.path.join(self.opt.out_path, 'log_files', 'train_log.txt'), mode) 
        f.write(msg)
        f.close()
             
    def log_epoch(self):  #在data/opts/log_files/train_log.txt文件中记录了训练的损失和测试的损失、TPR@FPR
        """ Epoch results log string"""
        out_train = 'Train: '
        out_test = 'Test:  '
        loss_i_string = 'Loss: {:.5f}\t'.format(self.train_loss.avg)
        out_train += loss_i_string
        loss_i_string = 'Loss: {:.5f}\t'.format(self.test_loss.avg)
        out_test += loss_i_string
            
        out_test+='\nTest:  '
        metrics_i_string = 'TPR@FPR=10-2: {:.4f}\t'.format(self.test_metrics.get_tpr(0.01))
        metrics_i_string += 'TPR@FPR=10-3: {:.4f}\t'.format(self.test_metrics.get_tpr(0.001))
        metrics_i_string += 'TPR@FPR=10-4: {:.4f}\t'.format(self.test_metrics.get_tpr(0.0001))
        out_test += metrics_i_string
        
        tn, fp, fn, tp,apcer,npcer,acer = self.test_metrics.get_apcer()
        out_test+='\nTest:  '
        matrix_i_string = 'TN:{}  FP:{}  FN:{}  TP:{}  APCER:{:.6f}  NPCER:{:.6f}  ACER:{:.8f}'.format(tn, fp, fn, tp,apcer,npcer,acer)
        out_test += matrix_i_string
        
            
        is_best = 'Best ' if self.best_epoch else ''
        out_res = is_best+'Epoch {} results:\n'.format(self.epoch)+out_train+'\n'+out_test+'\n'
        
        print(out_res)
        self.log_msg(out_res)

    def vislog_epoch(self):
        x_value = self.epoch
        self.vis.line([self.train_loss.avg], [x_value], 
                        name='train_loss', 
                        win='epoch_losses', 
                        update='append')
        self.vis.line([self.test_loss.avg], [x_value], 
                        name='val_loss', 
                        win='epoch_losses', 
                        update='append')
        self.vis.update_window_opts(win='epoch_losses', opts=self.vis_epochloss_opts)


        self.vis.line([self.test_metrics.get_tpr(0.01)], [x_value], 
                        name='tpr@fpr10-2', 
                        win='val_tpr', 
                        update='append')
        self.vis.line([self.test_metrics.get_tpr(0.001)], [x_value], 
                        name='tpr@fpr10-3', 
                        win='val_tpr', 
                        update='append')
        self.vis.line([self.test_metrics.get_tpr(0.0001)], [x_value], 
                        name='tpr@fpr10-4', 
                        win='val_tpr', 
                        update='append')
        self.vis.update_window_opts(win='val_tpr', opts=self.vis_tpr_opts)
                    
    def create_state(self):
        self.state = {       # Params to be saved in checkpoint
                'epoch' : self.epoch,
                'model_state_dict' : self.model.state_dict(),   #保存模型的参数
                'classifier_state_dict': self.classifier.state_dict(),
                'best_test_loss' : self.best_test_loss,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
            }
    
    def save_state(self):     #opt.log_checkpoint=1，为0时将保存断点
        save_name = '{}_best.pth.tar'.format(self.epoch) if self.best_epoch else\
            '{}.pth.tar'.format(self.epoch)
        if self.opt.log_checkpoint == 0:
                self.save_checkpoint('checkpoint.pth')
        else:
            if (self.epoch % self.opt.log_checkpoint == 0):
                self.save_checkpoint(save_name) 
                  
    def save_checkpoint(self, filename):     # Save model to task_name/checkpoints/filename.pth
        fin_path = os.path.join(self.opt.out_path,'checkpoints', filename)
        torch.save(self.state, fin_path)
        #if self.best_epoch:
            #best_fin_path = os.path.join(self.opt.out_path, 'checkpoints', 'model_best.pth')
            #torch.save(self.state, best_fin_path)
           

    def load_checkpoint(self):                            # Load current checkpoint if exists
        fin_path = os.path.join(self.opt.out_path,'checkpoints',self.opt.resume)
        print('fin_path:',fin_path)
        if os.path.isfile(fin_path):
            print("=> loading checkpoint '{}'".format(fin_path))
            checkpoint = torch.load(fin_path, map_location=lambda storage, loc: storage)
            #self.epoch = checkpoint['epoch'] + 1
            #self.best_test_loss = checkpoint['best_test_loss']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            print("=> loaded checkpoint '{}' (epoch {})".format(self.opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.opt.resume))

        #if os.path.isfile(self.visdom_log_file):
                #self.vis.replay_log(log_filename=self.visdom_log_file)
            
