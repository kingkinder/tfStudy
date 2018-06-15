# import os
# import sys
# root_path = os.path.abspath("./")
# if root_path not in sys.path:
    # sys.path.append(root_path)

import pandas as pd
import numpy as np
import math
from pandas import Series,DataFrame
from sklearn import preprocessing  #数据预处理包
import pickle as pkl
import h5py  
import pandas.core.algorithms as algos
import util.data_clear as cl

#提取有效因子模块   
#-------------------简单因子提取------------------
def simplefactor(data):
	
	ma20= data['close'].ewm(span=19.7).mean()
	ma10 = data['close'].ewm(span=10.39).mean()
	max=(ma10-ma20)/ma20
	mal= data['low']/ma10-1
	mah= data['high']/ma10-1
	datanew=data[['rtn','volumn','OI']].copy()  
	datanew['mal']=list(mal)
	datanew['mah']=list(mah)
	datanew['max']=list(max)
	datanew=simplefactoradd(datanew,1)
	return datanew

	#因子增加天数因子，用到过去num日的数据做决策，num>=1
def simplefactoradd(data,num,i=1):
	
	while i<=num:
		if i==1:
			data['rtn'+str(i)]=data['rtn'].shift(1)
			data['volumn'+str(i)]=data['volumn'].shift(1)
			data['OI'+str(i)]=data['OI'].shift(1)
		else:
			data['rtn'+str(i)]=data['rtn'+str(i-1)].shift(1)
			data['volumn'+str(i)]=data['volumn'+str(i-1)].shift(1)
			data['OI'+str(i)]=data['OI'+str(i-1)].shift(1)
		i=i+1
		simplefactoradd(data,num,i)
	data=data.dropna()
	return data  
	
	#增加自定义因子
def factoradd(data):
	
	while i<=num:
		if i==1:
			data['rtn'+str(i)]=data['rtn'].shift(1)
			data['volumn'+str(i)]=data['volumn'].shift(1)
			data['OI'+str(i)]=data['OI'].shift(1)
		else:
			data['rtn'+str(i)]=data['rtn'+str(i-1)].shift(1)
			data['volumn'+str(i)]=data['volumn'+str(i-1)].shift(1)
			data['OI'+str(i)]=data['OI'+str(i-1)].shift(1)
		i=i+1
		simplefactoradd(data,num,i)
	data=data.dropna()
	return data  	

def test(path='rb_3min.csv'):	
	frame=cl.datareadtime1(path)
	data=cl.dataprepareorgNtag(frame)
	return data
	
def getwight(num):
	dd=np.arange(num)+1
	weights =dd/np.sum(dd)
	

#-----------------衍生因子提取------------------









	


	
	
	
	





	
