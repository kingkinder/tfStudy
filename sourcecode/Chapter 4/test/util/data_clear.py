import pandas as pd
import numpy as np
import math
from pandas import Series,DataFrame
from sklearn import preprocessing  #数据预处理包
import pickle as pkl
import h5py  
import pandas.core.algorithms as algos

#import scipy.io as sio    
#import matplotlib.pyplot as plt    
#matpath='.\data_mat\data.mat' 

#加载mat测试数据
def loadTestmat(matpath='..\..\data_mat\datasimple.mat'):
	data = h5py.File(matpath,'r')  
	xtrain=data['input_train'].value
	ytrain=data['output_train'].value
	xtest=data['input_test'].value
	ytest=data['output_test'].value
	return xtrain,ytrain,xtest,ytest

def loadFutTestmat(matpath='..\..\data_mat\datasingle.mat'):
	data = h5py.File(matpath,'r')  
	xtrain=data['input_train'].value
	ytrain=data['output_train'].value
	xtest=data['input_test'].value
	ytest=data['output_test'].value
	return xtrain,ytrain,xtest,ytest	
	
def loadFlowTestmat(matpath='..\..\data_mat\dataflow.mat'):
	data = h5py.File(matpath,'r')  
	datax=data['datax'].value
	datay=data['datay'].value
	
	return datax,datay
	
#--------------从文件加载数据--------
#加载测试数据
def loadTest(path,days,num):
	df = pd.read_excel(path)
	dataset=df[['rtn','volumn','OI']].values
	datatime=df['time'].values
	datatag=gettag(dataset,days)#根据预测日期打标签
	#dataset=cl.dataCtime(datatag,num)#根据输入数据日期分割
	datatrain=dataCtime(datatag,num)
	datatime1=datatime[days-1:len(datatime)-num]
	return datatrain,datatime1

#从文件读取数据，包含某一列名称为time
def datareadtime(path):
	df = pd.read_excel(path)
	df.index = df['time'].tolist()
	return df

#从文件读取数据，其中第一列是日期time	
def datareadtime1(path='rbpro.csv'):
	s=path.split('.')[-1]
	if s=='xlsx' or s=='xls':
		dataorg=pd.read_excel(path,index_col=0)
		dataorg.index=pd.to_datetime(dataorg.index)
	elif s=='h5':
		dataorg=pd.read_hdf(path)
		#dataorg.index=pd.to_datetime(dataorg.index)
	else:
		dataorg=pd.read_csv(path,index_col=0)
		dataorg.index=pd.to_datetime(dataorg.index)
	return dataorg
	
#加载数据,做简单的归一化处理，不做时间切片
def loadorgdata(path):
	data=datareadtime1(path)
	frame=dataprepare(data)
	return frame
#----------以上是数据文件读取模块-----------		



#--------------文件保存---------------------
def savedata(data,path):
	s=path.split('.')[-1]
	if s=='xlsx' or s=='xls':
		pd.to_excel(path)
		
	elif s=='h5':
		#dataorg=pd.to_hdf('path', 'df')
		h5 = pd.HDFStore(path,'w')
		h5['data'] = data
		h5.close()
	elif s=='csv':
		data.to_csv(path)
	return 
#------------以上是文件保存模块-----------------	



#----------交易数据打标签------
def datagettag(data,days):#处理数据，不做归一化
	#frame=data.ix[:,['rtn','volumn','OI','high','low','close']]
	datatag=gettagpd(data.ix[:,['rtn']],days)
	#dataset=frame.values
	frame=data.copy()
	#datatest,scaler=prossZScore(dataset)  #数据归一化处理
	#frame[:]=dataset
	frame['tag']=datatag['tag']
	return frame

#----------交易数据预处理和打标签------
def dataprepareorg(data,days):#处理数据，不做归一化
	frame=data.ix[:,['rtn','volumn','OI','high','low','close']]
	datatag=gettagpd(frame.ix[:,['rtn']],days)
	#dataset=frame.values
	frame=dataprosspd(frame)  #数据预处理
	#datatest,scaler=prossZScore(dataset)  #数据归一化处理
	#frame[:]=dataset
	frame['tag']=datatag['tag']
	return frame	
	
#----------交易数据预处理不打标签------
def dataprepareorgNtag(data):#处理数据，不做归一化
	frame=data.ix[:,['rtn','volumn','OI','high','low','close']]
	
	#dataset=frame.values
	frame=dataprosspd(frame)  #数据预处理
	#datatest,scaler=prossZScore(dataset)  #数据归一化处理
	#frame[:]=dataset
	return frame

#处理数据，做归一化	
def dataprepare(data,days):
	frame=data.ix[:,['rtn','volumn','OI']]
	datatag=gettagpd(frame.ix[:,['rtn']],days)
	dataset=frame.values
	dataset=datapross(dataset)  #数据预处理
	datatest,scaler=prossZScore(dataset)  #数据归一化处理
	frame[:]=datatest
	frame['tag']=datatag['tag']
	return frame


	
def dataprosspd(data):  #数据预处理
	#datanew=np.zeros_like(data)
	datanew=data.ix[:,['OI']].values
	datac=datanew
	for i in range(1,len(datanew)):
		datac[i]=datanew[i]-datanew[i-1]
	data.ix[:,['OI']]=	datac
	return data

def datapross(data):  #数据预处理
	datanew=np.zeros_like(data)
	datanew[0,:]=data[0,:]
	datanew[0,2]=0
	for i in range(1,len(data)):
		datanew[i,0]=data[i,0]
		datanew[i,1]=data[i,1]
		datanew[i,2]=data[i,2]-data[i-1,2]
	return datanew

#----------交易数据预处理完成------

#--------------数据搜索与索引,时间序列处理-----------------	
#根据指定日期选择相应长度的数据
def getdatabylength(data,time,length):	
    index=data.index
    id=index[index>=time].min()
    sor=np.flatnonzero(index==id)[0]
    datenew=data[sor-length:sor+1]
    return datenew
	
#指定日期向后选数据
def getdatabylengthback(data,time,length):	
    index=data.index
    id=index[index>=time].min()
    sor=np.flatnonzero(index==id)[0]
    if length>=(len(index)-sor):length=length-sor
    datenew=data[sor:sor+length]
    return datenew	

#指定日期向前选数据，日期的正向
def getdatabylengthforward(data,time,length):	
    index=data.index
    id=index[index>=time].min()
    sor=np.flatnonzero(index==id)[0]
    if length>=(len(index)-sor):length=len(index)-sor
    datenew=data[sor:sor+length]
    return datenew		

#指定日期向后选数据，日期的反向
def getdatabylengthbackwards(data,time,length):	
    index=data.index
    id=index[index>=time].min()
    sor=np.flatnonzero(index==id)[0]
    if length>=(len(index)-sor):length=length-sor
    datenew=data[sor:sor+length]
    return datenew		
	
#指定日期向后选数据，切割数据
def datacutBytime(data,time,shift=0):	
    index=data.index
    id=index[index>=time].min()
    sor=np.flatnonzero(index==id)[0]-shift
    datab=data[:sor]
    dataf=data[sor:]
    return datab,dataf	
	

	
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)	
	
	
	
	


#---------------数据归一化----------------------
#使用sklearn.preprocessing.StandardScaler类，使用该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据。
#Z-Score
def prossZScore(X):
	scaler = preprocessing.StandardScaler().fit(X)
	data=scaler.transform(X) 
	storeDataPro(scaler,'proZscore.pkl')
	return data,scaler
	
def prossZScoreR(X,scaler):
	#pkl_file = open(path, 'rb')
	#scaler=pkl.load(pkl_file)
	#pkl_file.close()
	data=scaler.transform(X) 
	 
	return data
	
def storeDataPro(datap,name):
	# 创建HDF5文件
	file = open(name,'wb')
# 写入
	pkl.dump(datap, file)
	
# 。。。。。。。。。
	file.close()
	return

#-------------下面是数据离散化方案----------------------
def discut(data,bins):
	a=np.hstack((bins,float("inf")))
	#index=np.arange(a.size)
	m,n=data.shape
	newdata=np.random.rand(m,n)
	for i in range(m):
		for j in range(n):
			newdata[i,j]=np.min(np.where(data[i,j]<a))
	
	return newdata	
#--------------数据离散化1分位数--------------------------
def prossDisq(dataSet,num):  
	m,n = dataSet.shape      #获取数据集行列（样本数和特征数)  
	disMat = np.zeros(m*n).reshape(m,n)  #初始化离散化数据集 
	datarand=np.random.randn(100000)#正态分布
	bins = algos.quantile(np.unique(datarand), np.linspace(0, 1, num))#分类切片 
	disMat=discut(dataSet,bins)
	# for i in range(n):    #由于最后一列为类别，因此遍历前n-1列，即遍历特征列  
		# data1=dataSet[:,i]
		# datasp= pd.cut(data1,bins,precision=8)
		# lable=datasp.codes
		# disMat[:,i]=lable
		
	return disMat,bins
#--------------数据离散化1分位数恢复--------------------------
def prossDisqr(dataSet,bins):  
	
	disMat=discut(dataSet,bins)
	# for i in range(n):    #由于最后一列为类别，因此遍历前n-1列，即遍历特征列  
		# data1=dataSet[:,i]
		# datasp= pd.cut(data1,bins,precision=8)
		# lable=datasp.codes
		# disMat[:,i]=lable
		
	return disMat
	

#--------------数据离散化2等距离--------------------------
def dataDiscretize(dataSet):  
	m,n = shape(dataSet)    #获取数据集行列（样本数和特征数)  
	disMat = tile([0],shape(dataSet))  #初始化离散化数据集  
	for i in range(n-1):    #由于最后一列为类别，因此遍历前n-1列，即遍历特征列  
		x = [l[i] for l in dataSet] #获取第i+1特征向量  
		y = pd.cut(x,10,labels=[0,1,2,3,4,5,6,7,8,9])   #调用qcut函数，将特征离散化为10类，可根据自己需求更改离散化种类  
		for k in range(n):  #将离散化值传入离散化数据集  
			disMat[k][i] = y[k]      
	return disMat 



#--------------pandas数据离散化--------------------------	
def dataDisNum(dataSet,Num=8):  
	data1=dataSet.ix[:,:-1].values
	datas,scaler=prossZScore(data1) #归一化,------------需要保存
	datadis,bins=prossDisq(datas,Num) #离散化，分为10个类别------------需要保存
	dataSet.ix[:,:-1]=datadis
	return dataSet,scaler,bins
	
#--------------pandas数据重离散化--------------------------	
def dataDisNumR(dataSet,scaler,bins):  
	data=dataSet.values
	datas=prossZScoreR(data,scaler) #归一化,------------需要保存
	datadis=prossDisqr(datas,bins) #离散化，分为10个类别------------需要保存
	dataset=dataSet.copy()
	dataset.ix[:,:]=datadis
	return dataset








	
def mean(numbers):
	return sum(numbers)/float(len(numbers))	
	
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)


#-------------数据日期拼接处理---------	
def dataCtime(data,num):
	if num==1:
		return data
	datanew=np.zeros([len(data)-num+1,len(data[0])*num])
	#print(datanew.shape)
	
	datas=data
	#print(datas.shape)
	#tag=data[:,len(data[0])-1]
	for i in range(num-1,len(datas)):
		rows=datas[i]
		for j in range(1,num):
			rows1=sum(datas[i-j:i,:])
			rows=np.hstack((rows,rows1))
			
		#print(rows.shape)
		#print(tag[i])
		#rows=np.hstack((rows,tag[i]))
		datanew[i-num+1,:]=rows
		# datanew[i,1]=data[i,1]
		# datanew[i,2]=data[i,2]-data[i-1,1]
	#datanew=np.hstack((datanew,tag))
	return datanew

#数据按照天数处理，做数据叠加
def dataCtimepd(pdata,num):
	data=pdata.ix[:,0:-1].values
	tag=pdata.ix[:,-1].values
	
	if num==1:
		return pdata
	datanew=np.zeros([len(data)-num+1,len(data[0])*num])
	#print(datanew.shape)
	
	datas=data
	#print(datas.shape)
	#tag=data[:,len(data[0])-1]
	for i in range(num-1,len(datas)):
		rows=datas[i]
		for j in range(1,num):
			rows1=sum(datas[i-j:i,:])
			rows=np.hstack((rows,rows1))
			
		#print(rows.shape)
		#print(tag[i])
		#rows=np.hstack((rows,tag[i]))
		datanew[i-num+1,:]=rows
		# datanew[i,1]=data[i,1]
		# datanew[i,2]=data[i,2]-data[i-1,1]
	#datanew=np.hstack((datanew,tag))
	index=pdata.index[num-1:]
	tag=tag[num-1:]
	tag=tag.reshape(len(tag),1)
	datanew=np.hstack((datanew,tag))
	datanew=pd.DataFrame(datanew)
	datanew.index=index
	ss=np.arange(len(datanew.columns))
	#print(datanew.shape)
	c=list()
	for cs in ss:c.append(str(cs))
	#columnscolumnscolumnsprint(c)
	datanew.columns=c
	return datanew	
#------------------以上是日期处理------	
	
	
def gettag(data,days):
	
	stdv=stdev(data[:,0])
	#print(stdv)
	newdata=[]
	tga=True
	for i in range(0,len(data)-days):
		tag=2
		if sum(data[i+1:i+days+1,0])>1.6*stdv:
			tag=3
		elif sum(data[i+1:i+days+1,0])<-1.6*stdv:
			tag=1
		 
		datarow=np.hstack((data[i,:],tag))
		if tga:
			newdata=datarow
			tga=False
		else:
			newdata=np.vstack((newdata,datarow))
		
	return newdata
			

def gettagday(data,days):
	if  len(data.shape)==1:
	 data=data.reshape(len(data),1)
	stdv=stdev(data[:,0])
	method=1
	if method==1:
	  Threshold=1.2*stdv
	else:
	 Threshold=0.003
	#print(stdv)
	newdata=[]
	tga=True
	for i in range(0,len(data)-days):
		tag=2
		s=i+1
		while(s<i+days+1):
			if sum(data[i+1:s,0])>Threshold:
			 tag=3
			 break
			elif sum(data[i+1:s,0])<Threshold*-1:
			  tag=1
			  break
			s=s+1
		
		# if sum(data[i+1:i+days+1,0])>Threshold:
			# tag=3
		# elif sum(data[i+1:i+days+1,0])<Threshold*-1:
			# tag=1
		 
		datarow=np.hstack((data[i,:],tag))
		if tga:
			newdata=datarow
			tga=False
		else:
			newdata=np.vstack((newdata,datarow))
		
	return newdata			

def gettagpd(data,days):
	rtn=data['rtn']
	rtntag=pd.DataFrame(rtn)
	#print(rtntag.tail())
	rtntag['tag']=2
	#print(rtntag.tail())
	rvalue=rtn.values
	tag=gettagday(rvalue,days)
	
	#print(tag.shape)
	#print(rtntag.shape)
	#print(rtntag.head(2))
	rtntag.ix[:-1*days,:]=tag
	
	#rtntag[days:]=tag
	return rtntag
	
	
	
	
def cleardata(data,num):

	datatrain=dataCtime(data,num)
	return datatrain

#获取时间序列
def gettimelist(timeindex,timestart,recycle):
	timeindex=timeindex.sort_values()
	id=timeindex[timeindex>=timestart].min()
	sor=np.flatnonzero(timeindex==id)[0]
	timelist=timeindex.tolist()
	timelistnew=[]
	time1= timelist[sor]
	ll=len(timeindex)
	while(time1<timelist[-1]):
		timelistnew.append(time1)
		sor=sor+recycle
		if sor>=ll-1:
			sor=ll-1
			#timelistnew.append(time1)
			break
		time1= timelist[sor]
		
	return timelistnew
	
	




	
#-----------------数据切割处理(样本内外)-----------------	
#对数据进行切分	，按照比例切分为样本内和样本外
def datacultbyratio(data,ratio=0.9):
	data=data.sort_index()
	index=int(data.index.size*ratio)
	datatrain=data[:index]
	datatest=data[index+1:]
	#print('T')
	xtrain=datatrain.ix[:,0:-1].values
	ytrain=datatrain.ix[:,-1].values
	xtest=datatest.ix[:,0:-1].values
	ytest=datatest.ix[:,-1].values
	return xtrain,ytrain,xtest,ytest

#根据指定日期、向前长度，向后长度，重平衡周期来切割样本内外
def datacultsynpd(data,time,days,lback=5000,lforward=500):
	data1=getdatabylength(data,time,lback)
	data2=getdatabylengthback(data,time,lforward)
	lforward=data2.shape[0]
	datanew=data1.append(data2)
	datanew=datanew.drop_duplicates()
	datacl=dataCtimepd(datanew,days)
	ratio=lforward/(lback+lforward)
	xtrain,ytrain,xtest,ytest=datacultbyratio(datacl,ratio)
	return xtrain,ytrain,xtest,ytest

#对数据进行切分	，按照给定时间点切分为样本内和样本外
def datacultbytime(data,time='2018-1-1'):
	index=data.index
	id=index[index>=time].min()
	sor=np.flatnonzero(index==id)[0]
	datatrain=data[:sor]
	datatest=data[sor:]
	#print('T')
	xtrain=datatrain.ix[:,0:-1].values
	ytrain=datatrain.ix[:,-1].values
	xtest=datatest.ix[:,0:-1].values
	ytest=datatest.ix[:,-1].values
	return xtrain,ytrain,xtest,ytest



#-----------以上是交易数据切割样本内外------	
