import numpy as np
from sklearn.metrics import roc_curve, accuracy_score,confusion_matrix
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
        
class ROCMeter(object):
    """Compute TPR with fixed FPR"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.target = np.ones(0)
        self.output = np.ones(0)
        self.oup=np.empty([0,2])

    def update(self, target, output):   #output:是每个预测值对应的阳性概率
        # If we use cross-entropy   
        #print('output.shape',output.shape)     #output的形状是32*2
        #oup = torch.tensor(output)
        oupp=output
        self.oup=np.append(self.oup,oupp,axis=0)
        if len(output.shape) > 1 and output.shape[1] > 1:
            output = output[:,1]  #取第二列 正的概率
        elif len(output.shape) > 1 and output.shape[1] == 1:
            output = output[:,0]
        self.target = np.hstack([self.target, target])
        self.output = np.hstack([self.output, output])

    def get_tpr(self, fixed_fpr):
        fpr, tpr, thr = roc_curve(self.target, self.output)       #thr为阈值，不同的阈值对应不同的fpr和tpr
        tpr_filtered = tpr[fpr <= fixed_fpr]          #筛选出满足条件（fpr<给定值）的tpr
        if len(tpr_filtered) == 0:
            return 0.0
        return tpr_filtered[-1]      #返回tpr_filtered列表中的最后一个元素

    def get_accuracy(self, thr=0.5):
        acc = accuracy_score(self.target,
                             self.output >= thr)
        return acc
    
    def get_apcer(self):                
        
        oup=torch.tensor(self.oup)   #转换为张量
        #print('oup.shape:',oup.shape)
        #print('oup:',oup)
        _,predicted = torch.max(oup, 1)   #求张量中最大值的位置
        oupn=predicted.detach().numpy()   #将张量转换为numpy
        
             
        tn, fp, fn, tp = confusion_matrix(self.target, oupn).ravel()
        apcer = fp/(tn + fp)    #误识率
        npcer = fn/(fn + tp)     #误拒率
        acer = (apcer + npcer)/2
        return tn, fp, fn, tp,apcer,npcer,acer
    
    
    

    def get_top_hard_examples(self, top_n=10):
        diff_arr = np.abs(self.target - self.output)
        hard_indexes = np.argsort(diff_arr)[::-1]   #argsort(x)函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        hard_indexes = hard_indexes[:top_n]
        return hard_indexes, self.target[hard_indexes], self.output[hard_indexes]
        
       