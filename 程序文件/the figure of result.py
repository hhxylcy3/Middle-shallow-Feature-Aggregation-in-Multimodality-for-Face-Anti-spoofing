'''绘制TPR@FPR=10-4:的比较图'''

import re
import matplotlib.pyplot as plt
import numpy as np

#f0=open('11.txt','r')
f1=open('126242.txt','r')
f2=open('136242.txt','r')
f3=open('146242.txt','r')
f4=open('156242.txt','r')

def readacer(f):
    f1t=f.readlines()
    acer=[]
    for j in range(len(f1t)): 
        s=f1t[j]
        if re.findall( 'TPR@FPR=10-4:',s):
            ss = s.split(' ')
            ls=ss[-1]          #提取ACER:的数值
            ls=ls.replace( 'TPR@FPR=10-4:','')    #删除字符串ACER
            ls=ls.replace('\n','')     #删除回车符    
            ls=float(ls)
            #ls="%.2f%%" % (ls * 100)    #将数据用百分比格式输出
            acer.append(ls)
    return acer
acer1=readacer(f1)
acer2=readacer(f2)
acer3=readacer(f3)
acer4=readacer(f4)
#acer=readacer(f0)
#print(acer)  

#marker=['o','v','>','1','2','3','4']
x=np.linspace(0,149,num=150)
#y=[acer1,acer2,acer3,acer4]
#plt.ylim((0.0009,0.9000))
plt.plot(x,acer1,'ro-',label='times:2')
plt.plot(x,acer2,'gv-',label='times:3')
plt.plot(x,acer3,'b1-',label='times:4')
plt.plot(x,acer1,'c2-',label='times:5')


plt.xlabel('Epochs',fontsize=14)
plt.ylabel('TPR@FPR=10-4',fontsize=14)
plt.legend(loc='lower left')

plt.show()


'''ACER比较图'''


import re
import matplotlib.pyplot as plt
import numpy as np

f1=open('126242.txt','r')
f2=open('136242.txt','r')
f3=open('146242.txt','r')
f4=open('156242.txt','r')

def readacer(f):
    f1t=f.readlines()
    acer=[]
    for j in range(len(f1t)): 
        s=f1t[j]
        if re.findall( 'ACER:',s):
            ss = s.split(' ')
            ls=ss[-1]          #提取ACER:的数值
            ls=ls.replace( 'ACER:','')    #删除字符串ACER
            ls=ls.replace('\n','')     #删除回车符    
            ls=float(ls)
            #ls="%.2f%%" % (ls * 100)    #将数据用百分比格式输出
            acer.append(ls)
    return acer
acer1=readacer(f1)
acer2=readacer(f2)
acer3=readacer(f3)
acer4=readacer(f4)

#print(acer1)  

#marker=['o','v','>','1','2','3','4']
x=np.linspace(0,149,num=150)
#y=[acer1,acer2,acer3,acer4]
#plt.ylim((0.0009,0.9000))
plt.plot(x,acer1,'ro-',label='times:2')
plt.plot(x,acer2,'gv-',label='times:3')
plt.plot(x,acer3,'b1-',label='times:4')
plt.plot(x,acer1,'c2-',label='times:5')


plt.xlabel('Epochs',fontsize=14)
plt.ylabel('ACER',fontsize=14)
plt.legend(loc='upper left')

plt.show()


