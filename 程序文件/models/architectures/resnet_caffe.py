import torch.nn as nn  # type: ignore
import torch  # type: ignore
import torch.nn.init  # type: ignore
import math  # type: ignore
#import models

use_relu = False
use_bn = True

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_dw(inp, oup,stride,expand_ratio):
    hidden_dim = round(inp * expand_ratio)
    if expand_ratio == 1:
        conv_dw = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
    else:
        conv_dw = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
    return conv_dw


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    global use_bn
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=not use_bn)

class NSELayer(nn.Module):
    def __init__(self, kernel = 3):
        super(NSELayer, self).__init__()
        self.conv1 = conv_dw(2, 1,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y=self.sigmoid(y)        
        return x * y

class CSAttention(nn.Module):   #注意力机制（空间和通道注意力，将特征图的尺寸压缩为1*1）
    def __init__(self, input_channel, bottleneck_channel,kernel_size=3):   #input_size指输入通道数，bottleneck_size指中间转换通道数（input_size//4）
        super(CSAttention, self).__init__()
        self.input_channel = input_channel
        self.bottleneck_channel = bottleneck_channel
        self.kernel_size = kernel_size

        self.se_fc1 = nn.Conv2d(self.input_channel, self.bottleneck_channel, kernel_size = 1)   #1*1卷积，改变通道数
        self.se_fc2 = nn.Conv2d(self.bottleneck_channel, self.input_channel, kernel_size = 1)
        self.conv1 = nn.Conv2d(2, 1, kernel_size = self.kernel_size, padding=self.kernel_size//2)
        self.se_fc3 = nn.Conv2d(self.input_channel*2, self.input_channel, kernel_size = 1)

    def forward(self, x):
        s_max = nn.functional.max_pool2d(x, x.size(2))      #尺寸为：b*c*1*1
        s_max = nn.functional.relu(self.se_fc1(s_max))     #尺寸为：b*c//4*1*1

        s_avg = nn.functional.avg_pool2d(x, x.size(2))         #尺寸为：b*c*1*1
        s_avg = nn.functional.relu(self.se_fc1(s_avg))

        s = s_max + s_avg
        s = torch.sigmoid(self.se_fc2(s))    #尺寸为：b*c*1*1
        xs=x*s
        
        c_max, _ = torch.max(x, dim=1, keepdim=True)  #尺寸为：b*1*h*w
        c_avg = torch.mean(x, dim=1, keepdim=True)

        c = torch.cat([c_avg, c_max], dim=1)   #尺寸为：b*2*h*w
        c = torch.sigmoid(self.conv1(c))     #尺寸为：b*1*h*w        
        xc = x*c
        xsc= torch.cat([xs,xc], dim=1) 

        x = torch.sigmoid(self.se_fc3(xsc))
        return x

    
    
def make_layer(exp_ratio,in_channel, out_channel, num_times,csa = False, nse = True,avgdown = True):
    layers = []
    for i in range(num_times):
        downsample = None
        if i == 0:
            if avgdown:
                downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                        nn.BatchNorm2d(in_channel),
                        nn.Conv2d(in_channel, out_channel , kernel_size=1, bias=False)
                        )
            layers.append(InvertedResidual(in_channel, out_channel, 2, exp_ratio, downsample = downsample))
        else:
            layers.append(InvertedResidual(in_channel, out_channel, 1, exp_ratio, downsample = downsample))
        in_channel = out_channel
    #if csa:
        #layers.append(CSAttention(in_channel, in_channel//4))
    if nse:
        layers.append(NSELayer())
    return nn.Sequential(*layers) 

def calculate_scale(data):
    if data.dim() == 2:
        scale = math.sqrt(3 / data.size(1))
    else:
        scale = math.sqrt(3 / (data.size(1) * data.size(2) * data.size(3)))
    return scale

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, downsample=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.downsample = downsample

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = conv_dw(inp, oup,stride,expand_ratio)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            if self.downsample is not None:
                return self.downsample(x) + self.conv(x)
            else:
                return self.conv(x)

class FeatherNet2(nn.Module):
    def __init__(self, csa = False, nse = True,avgdown=True):
        super(FeatherNet2, self).__init__()
        self.csa = csa
        self.nse = nse
      

        # building  layer
        self.conv1 = conv_bn(3, 16, 2)    #核为3的conv2d + batchnorm + relu6 ,channel: 3-32,size:56
        #make_layer参数：3表示扩展率，32表示输入通道数，48表示输出通道数；2表示循环次数
        self.layer1 = make_layer(1,16, 48,3,csa,nse,avgdown)   #channel: 16-48 size:28 
        self.max_pool = nn.MaxPool2d(2, stride=2)          #channel: 48-48  size:14  
        self.conv_dw =conv_dw(48, 128,2,1)     #channel: 48-128 size:7  
        
        self.layer2 = make_layer(6,48, 64,2,csa,nse,avgdown)   # channel: 48-64  size:14 
        
        self.layer3 = make_layer(4,64, 128, 3, csa,nse,avgdown)   # channel: 64-128  size:7
        self.conv1x1_2 = conv_1x1_bn(256, 128)   #channel: 256-128  size:7  
        
        
                
        #self.conv_cat1 = conv_1x1_bn(256, 128)        # channel: 128-64  size:7 
        
#         building last several layers  两种方案：layer1和layer3拼接后，直接深度卷积到512维(self.DW2)；另一种方法是拼接后做一次1*1卷积，降维到128，再做一次深度卷积，升维到512（self.conv_cat1,self.DW1)      
        #self.DW1 = conv_dw(128, 512,stride=2,expand_ratio=1)   #channel: 64-256  size:3  
       
        #self.DW2 = conv_dw(256, 512,stride=2,expand_ratio=1)   #channel: 256-512  size:3 
            
        
        self.CSA = CSAttention(128, 64)
        self.NSE = NSELayer()
        #self.main_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)             #channel: 3-16  size:56 
        x_layer1 = self.layer1(x)      #channel: 16-48  size:28 
        x_layer11 = self.max_pool(x_layer1)     #channel: 48  size:14
        x_layer11 = self.conv_dw(x_layer11)           #channel: 48-128 size:7
        
        x_layer2 = self.layer2(x_layer1)      # channel: 48-64  size:14 
        #x_layer22 = torch.cat([x_layer11, x_layer2], dim=1)   #channel: 128  size:14 
        #x_layer22 = self.max_pool(x_layer22)     #channel: 128  size:7
        
        x_layer3 = self.layer3(x_layer2)        # channel: 64-128  size:7  
        x_layer33 = torch.cat([x_layer11, x_layer3], dim=1)   #channel: 256  size:7
        x_layer33 = self.conv1x1_2(x_layer33)           #channel: 256-128  size:7    
        
        #x_layer22 = self.conv_dw2(x_layer2)    #channel: 48-64  size:14 
        #x_layer12 = torch.cat([x_layer11, x_layer22], dim=1)       #channel: 128  size:14     
        #x_layer12 = self.conv_cat1(x_layer12)           # channel: 256-128  size:7 
        
        
        #x_layer = torch.cat([x_layer3, x_layer11], dim=1)       # channel: 256  size:7
        #x_layer = self.conv_cat1(x_layer)              # channel: 256-128  size:7 
        
        #x_layer = self.DW1(x_layer)        #512*3*3
        #x_layer = self.DW2(x_layer)    #512*3*3
        if self.csa:
            x_layer = self.CSA(x_layer33)     #128*7*7
        if self.nse:
            x_layer =self.NSE(x_layer33)
        #x_layer = self.main_avgpool(x_layer)    #512*1*1
        #x_layer = x_layer.view(x_layer.size(0), -1)    
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x_layer

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class FeatherNet22(nn.Module):
    def __init__(self,ex,rx,csa = False, nse = True,avgdown=True):
        super(FeatherNet22, self).__init__()
        self.csa = csa
        self.nse = nse
 
        # building  layer
        self.conv1 = conv_bn(3, 16, 2)    #核为3的conv2d + batchnorm + relu6 ,channel: 3-16,size:56
        #make_layer参数：3表示扩展率，32表示输入通道数，48表示输出通道数；3表示循环次数
        self.layer1 = make_layer(ex[0],16, 48,rx[0],csa,nse,avgdown)   #channel: 16-48 size:28 
        self.max_pool = nn.MaxPool2d(2, stride=2)          #channel: 48-48  size:14  
        self.conv_dw =conv_dw(48, 128,2,1)     #channel: 48-128 size:7  
        
        self.layer2 = make_layer(ex[1],48, 64,rx[1],csa,nse,avgdown)   # channel: 48-64  size:14 
        
        self.layer3 = make_layer(ex[2],64, 128, rx[2], csa,nse,avgdown)   # channel: 64-128  size:7
        self.conv1x1_2 = conv_1x1_bn(256, 128)   #channel: 256-128  size:7  
        #self.conv_cat1 = conv_1x1_bn(256, 128)        # channel: 128-64  size:7 
        
#         building last several layers  两种方案：layer1和layer3拼接后，直接深度卷积到512维(self.DW2)；另一种方法是拼接后做一次1*1卷积，降维到128，再做一次深度卷积，升维到512（self.conv_cat1,self.DW1)      
        #self.DW1 = conv_dw(128, 512,stride=2,expand_ratio=1)   #channel: 64-256  size:3  
       
        #self.DW2 = conv_dw(256, 512,stride=2,expand_ratio=1)   #channel: 256-512  size:3 
            
        
        #self.CSA = CSAttention(128, 64)
        self.NSE = NSELayer()
        #self.main_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)             #channel: 3-32  size:56 
        x_layer1 = self.layer1(x)      #channel: 32-48  size:28 
        x_layer11 = self.max_pool(x_layer1)     #channel: 48  size:14
        x_layer11 = self.conv_dw(x_layer11)           #channel: 48-128 size:7
        
        x_layer2 = self.layer2(x_layer1)      # channel: 48-64  size:14 
        #x_layer22 = torch.cat([x_layer11, x_layer2], dim=1)   #channel: 128  size:14 
        #x_layer22 = self.max_pool(x_layer22)     #channel: 128  size:7
        
        x_layer3 = self.layer3(x_layer2)        # channel: 64-128  size:7  
        x_layer33 = torch.cat([x_layer11, x_layer3], dim=1)   #channel: 256  size:7
        x_layer33 = self.conv1x1_2(x_layer33)           #channel: 256-128  size:7        
        
        
        #if self.csa:
            #x_layer = self.CSA(x_layer33)     #128*7*7
        if self.nse:
            x_layer =self.NSE(x_layer33)
        
        return x_layer

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
                
                
class FeatherNetSiam2(nn.Module):   #last_layers为真时，表示对最后的层进行全连接
    def __init__(self, csa = False, nse = True,avgdown=True,last_layers=False):
        super(FeatherNetSiam2, self).__init__()
        global use_bn
        self.use_bn = use_bn        
        self.inplanes = 128 * 3
        self.rgb_backbone = FeatherNet2()   #128*7*7
        self.depth_backbone = FeatherNet2()
        self.ir_backbone = FeatherNet2()
#make_layer的参数：exp_ratio,in_channel, out_channel, num_times,csa = False, nse = True,avgdown = True
        self.layer4 = make_layer(1,384, 512,2, csa = False, nse = True,avgdown = False)   #512*3*3
        #self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     #512*1*1

        self.last_layers = last_layers
        if self.last_layers:
            self.fc1 = nn.Linear(512, 128)
            self.fc2 = nn.Linear(128, 512)          
   

    def forward(self, x, y, z):
        x = self.rgb_backbone(x)
        y = self.depth_backbone(y)    #128*7*7
        z = self.ir_backbone(z)

        x = torch.cat((x,y,z), dim=1)    #384*7*7
        
        x = self.layer4(x)    #512*3*3
        #x = self.layer5(x)
        x = self.avgpool(x)   #512*1*1

        x = x.view(x.size(0), -1)

        if self.last_layers:
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)

        return x
class FeatherNetSiam22(nn.Module):   #last_layers为真时，表示对最后的层进行全连接
    def __init__(self,ex,rx, csa = False, nse = True,avgdown=True,last_layers=False):
        super(FeatherNetSiam22, self).__init__()
        global use_bn
        self.use_bn = use_bn        
        self.inplanes = 128 * 3        
        self.rgb_backbone = FeatherNet22(ex,rx,)   #128*7*7
        self.depth_backbone = FeatherNet22(ex,rx,)
        self.ir_backbone = FeatherNet22(ex,rx,)
#make_layer的参数：exp_ratio,in_channel, out_channel, num_times,csa = False, nse = True,avgdown = True
        self.layer4 = make_layer(ex[3],128*3, 512,rx[3], csa = False, nse = True,avgdown = False)   #512*3*3
        #self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     #512*1*1

        self.last_layers = last_layers
        if self.last_layers:
            self.fc1 = nn.Linear(512, 128)
            self.fc2 = nn.Linear(128, 512)          
   

    def forward(self, x, y, z):
        x = self.rgb_backbone(x)
        y = self.depth_backbone(y)    #128*7*7
        z = self.ir_backbone(z)

        x = torch.cat((x,y,z), dim=1)    #384*7*7
        
        x = self.layer4(x)    #512*3*3
        #x = self.layer5(x)
        x = self.avgpool(x)   #512*1*1

        x = x.view(x.size(0), -1)

        if self.last_layers:
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)

        return x
    


def FeatherNet_siam(pretrained=False,last_layers=False, **kwargs):  #ex是各层的扩展系数；rx是各层的循环次数
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FeatherNetSiam2(csa = False, nse = True,avgdown=True,last_layers=last_layers, **kwargs) #[3,4,6,3]
    #if pretrained:
        #weights = torch.load('D:\\lcy\\feathernet\\CVGNet\\mlfa\\data\\opts\\exp1FeatherNetCC\\fold1\\checkpoints\\116.pth', map_location='cpu')
    
    #model = FeatherNetSiam22(ex=[3,6,3,1],rx=[3,3,3,2],csa = False, nse = True,avgdown=True,last_layers=last_layers) #[3,4,6,3]
    #if pretrained:
        #if pretrained == 'cnn46_fp0':
            #weights = torch.load('/media2/a.parkin/codes/Liveness_challenge/models/pretrained/resnet_caffe_v2.pth')
            
        #elif pretrained == 'mlfa2021':
            #weights = torch.load('D:\\lcy\\feathernet\\CVGNet\\mlfa\\data\\opts\\exp1FeatherNetCC\\fold1\\checkpoints\\model_93.pth', map_location='cpu')
        
        #if last_layers:
            #state_dict = model.state_dict()
            #weights['fc1.weight'] = state_dict['fc1.weight']
            #weights['fc1.bias'] = state_dict['fc1.bias']
            #weights['fc2.weight'] = state_dict['fc2.weight']
            #weights['fc2.bias'] = state_dict['fc2.bias']
        
        #print(weights.keys())
        #model.load_state_dict({k.replace('module.',''):v for k,v in weights['model_state_dict'].items()})   #torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
    return model            
            
            

