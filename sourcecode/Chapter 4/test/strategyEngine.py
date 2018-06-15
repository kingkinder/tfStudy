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
from util.strategyEngine_base import strategyEngine_base
import util.data_clear as cl



class strategyEngine_sim(strategyEngine_base):
    def __init__(self):
        super(strategyEngine_sim, self).__init__() #调用父类
        self.scaler=None
        self.bins=None
        self.factornum=0
        #self._probs = probs

    def fit(self):
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(self._xtrain,self._ytrain)
        predict=clf.predict(self._xtest)
        core=ML.getscore(self._ytest, predict)
        self._clf =clf
        self._core=core
		#AdaBoost的关键：通过顺序的学习一些弱分类器（这些 弱分类器仅比随机好一点），然后通过加权投票得到最后的预测。
        from sklearn.ensemble  import AdaBoostClassifier   
        classifier =AdaBoostClassifier(n_estimators=100)     
        predict = classifier.fit(self._xtrain,self._ytrain).predict(self._xtest)
        core1=ML.getscore(self._ytest, predict)
        if core1>core:
            self._core=core1
            self._clf =classifier
		# GradientBoostingClassifier能够支持二分类和多分类。
        from sklearn.ensemble  import GradientBoostingClassifier  
        classifier =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        predict = classifier.fit(self._xtrain,self._ytrain).predict(self._xtest)
        core1=ML.getscore(self._ytest, predict)
        if core1>core:
            self._core=core1
            self._clf =classifier

        from sklearn.naive_bayes  import MultinomialNB
        classifier =MultinomialNB()
        predict = classifier.fit(self._xtrain,self._ytrain).predict(self._xtest)
        core1=ML.getscore(self._ytest, predict)
        if core1>core:
            self._core=core1
            self._clf =classifier

        from sklearn.naive_bayes  import BernoulliNB
        classifier =BernoulliNB()
        predict = classifier.fit(self._xtrain,self._ytrain).predict(self._xtest)
        core1=ML.getscore(self._ytest, predict)
        if core1>core:
            self._core=core1
            self._clf =classifier
        #print("best point is %.2f"%(self._core*100))
    def onInit(self,data):
		#"""策略装载数据"""
		#加载简单因子
        data=fc.simplefactor(data)

        self._data =data

    def trainbytime(self, time='2018-1-1'):
        # y_pred=self._clf.predict(x)
        # xtrain,ytrain,xtest,ytest =cl.datacultbytime(data,time)
        return time


    def trainbytimelenth(self, time='2018-1-1',lenback=11000,lenforward=500):
        #y_pred=self._clf.predict(x)
        #因子组合，复杂化
        #data=fc.simplefactoradd(self._data,1)
        #self.factornum=1
        #数据切割，训练用
        datab,dataf=cl.datacutBytime(self._data,time,shift=lenback)
        data,s=cl.datacutBytime(dataf,time)
        #数据打标签，做训练用
        data=cl.datagettag(data,10)
        #数据离散化
        data,scaler,bins=cl.dataDisNum(data,20)
        self.scaler=scaler
        self.bins=bins
        
        xtrain,ytrain,xtest,ytest=cl.datacultbyratio(data,0.9)
        self.feed_data(xtrain,ytrain,xtest,ytest)
        self.fit()
        self.retrain()
        probs=0.6
        _proba=self.predictprob(xtest)
        y_pred=ML.toNum(_proba,probs)
        ML.getscoreTrade(ytest,y_pred)
        #print(ML.score(ytest,y_pred))
    #根据最新数据做预测
    #data包含最原始的因子，close,high,low,rtn,OI(相对值),Vol
    def predictnow(self,data):
        #数据预处理，因子提取
        data=fc.simplefactor(data)
        #data=fc.simplefactoradd(data,num=self.factornum)
        data=cl.dataDisNumR(data,self.scaler,self.bins)
        datas=data.values
        prob=self.predictprob(datas)
        return prob[-1]

#根据回测时间点，指定回测数据长度，预测长度等数据，不选择最佳策略，返回预测结果，重写
    def backtestbytimesb(self,time,lenback=11000,lenforward=500,point=0.5):
        #y_pred=self._clf.predict(x)
        #因子组合，复杂化
        #data=fc.simplefactoradd(self._data,1)
       #self.factornum=1
        #数据切割，训练用
        datab,dataf=cl.datacutBytime(self._data,time,shift=lenback)
        
        #数据打标签，做训练用
        dataf=cl.datagettag(dataf,4)
        #数据离散化
        dataf,scaler,bins=cl.dataDisNum(dataf,9)
        self.scaler=scaler
        self.bins=bins

        data,s=cl.datacutBytime(dataf,time)
        xtrain,ytrain,xtest,ytest=cl.datacultbyratio(data,0.9)
        self.feed_data(xtrain,ytrain,xtest,ytest)
        
        self.fit()
        print('the best score is %.2f '%(100*self._core))
        self.retrain()
        #print(xtrain.shape)
        datanew=cl.getdatabylengthback(dataf,time,self.factornum)
        datanew1=cl.getdatabylengthforward(dataf,time,lenforward-1)
        datanew=datanew1.append(datanew)
        datanew=datanew.drop_duplicates()
        pout=self.NNpredict(datanew,self._clf,lenforward)
        return pout
#根据起始时间点和每次运算区间做出回测
    def backtestBytimelength(self,time,lenback=11000,lenforward=500,point=0.51):
        timelist=cl.gettimelist(self._data.index,time,lenforward)
        tag=False
        pout=[]
        for t in timelist:
            print('backtest at time: %s' %t)
            if tag:
                pout1=self.backtestbytimesb(t,lenback,lenforward)
                pout=pout.append(pout1)
            else:
                pout=self.backtestbytimesb(t,lenback,lenforward)
                tag=True
        #pout.to_csv('testpredict.csv')
        return pout


     
