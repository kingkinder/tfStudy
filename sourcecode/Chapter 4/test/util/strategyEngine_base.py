# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import h5py
import numpy as np
import pickle as pkl
import pandas as pd

from pandas import Series,DataFrame
#from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score#平均准确率
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import util.data_factors as fc
import util.ML_ultimate as ML
import util.data_clear as cl


class strategyEngine_base(object):
    def __init__(self, **kwargs):
        self._xtrain = self._ytrain = self._xtest = self._ytest = None
        self._clf = None
        self._core = None
        self._cat_counter = self._con_counter = None
        self._data = None
    def feed_data(self,xtrain,ytrain,xtest,ytest):
        self._xtrain = xtrain
        self._ytrain = ytrain
        self._xtest =  xtest
        self._ytest =  ytest
    def fit(self):
        """搜索策略（必须由用户继承实现）"""
        raise NotImplementedError
        #print("best point is %.2f"%(self._core*100))
    def onInit(self):
        """初始化策略（必须由用户继承实现）"""
        raise NotImplementedError
    def predict(self, x, **kwargs):
        y_pred=self._clf.predict(x)
        return y_pred
    def predictprob(self, x, **kwargs):
        y_pred=self._clf.predict_proba(x)
        return y_pred
    def retrain(self):
        #print(self._ytrain.shape)
        #print(self._ytest.shape)
        datax=np.vstack((self._xtrain,self._xtest))
        datay=np.vstack((self._ytrain.reshape(len(self._ytrain),1),self._ytest.reshape(len(self._ytest),1)))
        datay=datay.ravel()
        self._clf.fit(datax,datay)

    #策略预测	
    def NNpredict(self,data,NN,times):
        #pdata=cl.dataCtimepd(data,times)
        input=data.ix[:,:-1].values
        
        output=NN.predict_proba(input)
        
        outreal=data.ix[:,-1]
	
        pout=pd.concat([outreal, pd.DataFrame(outreal,columns=['down','stay','up'])])
        pout.ix[:,1:]=output
	#score=getscoreTrade(outreal,output)
        return pout

    #根据回测时间点，指定回测数据长度，预测长度等数据，不选择最佳策略，返回预测结果	
    def backtestbytimesb(self,time,lenback=11000,lenforward=500,point=0.5):
        data1=cl.getdatabylengthback(self._data,time,lenback)
        xtrain,ytrain,xtest,ytest=cl.datacultbyratio(data1,0.9)
        self.feed_data(xtrain,ytrain,xtest,ytest)
        self.fit()
        self.retrain()
        datanew=cl.getdatabylengthback(self._data,time,lenback)
        datanew1=cl.getdatabylengthforward(self._data,time,lenforward-1)
        datanew=datanew1.append(datanew)
        datanew=datanew.drop_duplicates()
        pout=ML.NNpredict(datanew,self._clf,lenforward)
        return pout


