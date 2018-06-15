# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import pickle as pkl
import pprint
import pandas as pd

import numpy as np
import util.ML_ultimate as ML
import util.data_clear as cl
import strategyEngine as ms

 

#策略训练
def testeg():
	path='T_15.csv'
	frame=cl.datareadtime1(path)
	data=cl.dataprepareorgNtag(frame)
	NN=ms.strategyEngine_sim()
	NN.onInit(data)
	NN.trainbytimelenth(time='2018-1-1',lenback=10000)
	#策略保存
	ML.storestrategy(NN,path='NB_T15.pkl')
	NN=NN._clf,path
	ML.storestrategy(NN,path='NB_T15SM.pkl')
#策略使用和预测	
def testPredic():

	NN=ML.loadstrategy(path='NB_T15.pkl')
	path='T_15.csv'
	frame=cl.datareadtime1(path)
	data=cl.dataprepareorgNtag(frame)
	#cl.savedata(data,'test2.csv')
	datanow=data[-30:]#选择最近的数据做判断
	predict=NN.predictnow(datanow)
	print(str(datanow.index[-1])+' 时刻预测后面行情走势为：')
	print('下跌概率为%.2f %%'%(100*predict[0]))
	print('横盘概率为%.2f %%'%(100*predict[1]))
	print('上涨概率为%.2f %%'%(100*predict[2]))
	
#策略回测	
def testbacktest():

	path='T_15.csv'
	frame=cl.datareadtime1(path)
	print('loaded data')
	data=cl.dataprepareorgNtag(frame)
	NN=ms.strategyEngine_sim()
	NN.onInit(data)
	pout=NN.backtestBytimelength(time='2017-1-1',lenback=5000,lenforward=300,point=0.51)
	pout.to_csv('testpredict.csv')
	print('predict has been saved')

	
#测试某一时间点分割为样本内外数据，测试策略效果	，简单分类的	
def test1dis(path='rb3dis1.csv'):#离散化,离散化分数更高
	
	data=cl.datareadtime1(path)
	xtrain,ytrain,xtest,ytest =cl.datacultbytime(data,time='2018-1-1')
	NN=ms.strategyEngine_sim()
	NN.feed_data(xtrain,ytrain,xtest,ytest)
	NN.fit()
	NN.retrain()
	probs=0.6
	# y_pred=NN.predict(xtest)
	_proba=NN.predictprob(xtest)
	y_pred=eg.toNum(_proba,probs)
	ML.getscoreTrade(ytest,y_pred)
	print(ML.score(ytest,y_pred))	
	
	
	
	
if __name__ == '__main__':
    #import time
	#testeg()
    testPredic()
	#testbacktest()