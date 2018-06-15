# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import h5py
import numpy as np
import pickle as pkl
import pandas as pd
import util.data_clear as cl
from pandas import Series,DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score#平均准确率
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools



def score(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    score=accuracy_score(y_true, y_pred)
    cnf_matrix=confusion_matrix(y_true,y_pred)
    class_names=getname(y_true)
    plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=False,title='Confusion matrix, without normalization')
    return score


#预测概率值转变为数字
def toNum(data,prob):
    s=[]
    for dd in data:
        if max(dd)>=prob:
            das=int(np.flatnonzero(dd>=prob))+1
        else:
            das=int(np.flatnonzero(dd==max(dd)))+1
        s.append(das)
    predict=np.array(s)
    return predict


#预测概率值转变为数字	
def probtoNum(data,prob):
    s=[]
    for dd in data:
        if max(dd)>=prob:
            das=int(np.flatnonzero(dd>=prob))+1
        else:
            das=int(np.flatnonzero(dd==max(dd)))+1
        s.append(das)
    predict=np.array(s)
    return predict
	
	
def getscoreTrade(y_true, y_pred):
	#from sklearn.metrics import accuracy_score
	#score=accuracy_score(y_true, y_pred)
	cnf_matrix=confusion_matrix(y_true,y_pred)
	t1=sum(cnf_matrix[0,:])#真实做空机会
	t2=sum(cnf_matrix[1,:])
	t3=sum(cnf_matrix[2,:])#真实做多机会
	p1=sum(cnf_matrix[:,0])#判断做空次数
	p2=sum(cnf_matrix[:,1])
	p3=sum(cnf_matrix[:,2])#判断做空次数
	P1T=cnf_matrix[0,0]#判断做空为正确次数
	P1N=cnf_matrix[1,0]#判断做空为震荡次数
	P1F=cnf_matrix[2,0]#判断做空实际上涨次数
	P3T=cnf_matrix[2,2]#判断做多为正确次数
	P3N=cnf_matrix[1,2]#判断做多为震荡次数
	P3F=cnf_matrix[0,2]#判断做多实际下跌次数
	if(p1<1):p1=1
	if(t1<1):t1=1
	p1tRate=(P1T-P1F+P1N*0.5)/p1;#判断做空正确概率
	# if p1tRate<0:
		# p1tRate=0;
	p1Hit=P1T/t1#做空机会命中率
	if(p3<1):p3=1
	if(t3<1):t3=1
	p3tRate=(P3T-P3F+P3N*0.5)/p3;#判断做多正确概率
	p3Hit=P3T/t3;#做多机会命中率
	print('做空正确概率:%.2f %%'%(100*p1tRate))
	print('做空命中概率:%.2f %%'%(100*p1Hit))
	print('做多正确概率:%.2f %%'%(100*p3tRate))
	print('做多命中概率:%.2f %%'%(100*p3Hit))
	score=(0.682*p1tRate+0.318*p1Hit)*0.5+(0.682*p3tRate+0.318*p3Hit)*0.5
	print('策略评分为:%.2f' % (score*100))
	return score	
	
	
	
def getscore(y_true, y_pred):
	#from sklearn.metrics import accuracy_score
	#score=accuracy_score(y_true, y_pred)
	cnf_matrix=confusion_matrix(y_true,y_pred)
	t1=sum(cnf_matrix[0,:])#真实做空机会
	t2=sum(cnf_matrix[1,:])
	t3=sum(cnf_matrix[2,:])#真实做多机会
	p1=sum(cnf_matrix[:,0])#判断做空次数
	p2=sum(cnf_matrix[:,1])
	p3=sum(cnf_matrix[:,2])#判断做空次数
	P1T=cnf_matrix[0,0]#判断做空为正确次数
	P1N=cnf_matrix[1,0]#判断做空为震荡次数
	P1F=cnf_matrix[2,0]#判断做空实际上涨次数
	P3T=cnf_matrix[2,2]#判断做多为正确次数
	P3N=cnf_matrix[1,2]#判断做多为震荡次数
	P3F=cnf_matrix[0,2]#判断做多实际下跌次数
	if(p1<1):p1=1
	if(t1<1):t1=1
	p1tRate=(P1T-P1F+P1N*0.5)/p1;#判断做空正确概率
	# if p1tRate<0:
		# p1tRate=0;
	p1Hit=P1T/t1#做空机会命中率
	if(p3<1):p3=1
	if(t3<1):t3=1
	p3tRate=(P3T-P3F+P3N*0.5)/p3;#判断做多正确概率
	p3Hit=P3T/t3;#做多机会命中率
	#print('做空正确概率:%.2f %%'%(100*p1tRate))
	#print('做空命中概率:%.2f %%'%(100*p1Hit))
	#print('做多正确概率:%.2f %%'%(100*p3tRate))
	#print('做多命中概率:%.2f %%'%(100*p3Hit))
	score=(0.682*p1tRate+0.318*p1Hit)*0.5+(0.682*p3tRate+0.318*p3Hit)*0.5
	#print('策略评分为:%.2f' % (score*100))
	return score	


#直接对策略评分，不用概率	
def getStrategescore(NN,xtest, ytest):
	ypredict=NN.predict(xtest)
	cnf_matrix=confusion_matrix(ytest,ypredict)
	t1=sum(cnf_matrix[0,:])#真实做空机会
	t2=sum(cnf_matrix[1,:])
	t3=sum(cnf_matrix[2,:])#真实做多机会
	p1=sum(cnf_matrix[:,0])#判断做空次数
	p2=sum(cnf_matrix[:,1])
	p3=sum(cnf_matrix[:,2])#判断做空次数
	P1T=cnf_matrix[0,0]#判断做空为正确次数
	P1F=cnf_matrix[2,0]#判断做空实际上涨次数
	P3T=cnf_matrix[2,2]#判断做多为正确次数
	P3F=cnf_matrix[0,2]#判断做多实际下跌次数
	if(p1<1):p1=1
	if(t1<1):t1=1
	if(p3<1):p3=1
	if(t3<1):t3=1
	p1tRate=(P1T-P1F)/p1;#判断做空正确概率
	# if p1tRate<0:
		# p1tRate=0;
	p1Hit=P1T/t1;#做空机会命中率
	p3tRate=(P3T-P3F)/p3;#判断做多正确概率
	p3Hit=P3T/t3;#做多机会命中率
	#print('做空正确概率:%.2f %%'%(100*p1tRate))
	#print('做空命中概率:%.2f %%'%(100*p1Hit))
	#print('做多正确概率:%.2f %%'%(100*p3tRate))
	#print('做多命中概率:%.2f %%'%(100*p3Hit))
	score=(0.682*p1tRate+0.318*p1Hit)*0.5+(0.682*p3tRate+0.318*p3Hit)*0.5
	#print('策略评分为:%.2f' % (score*100))
	return score	
	
#直接对策略评分，用上概率	
def getStrategescorepron(NN,xtest, ytest,probability=0.6):
	#from sklearn.metrics import accuracy_score
	#score=accuracy_score(y_true, y_pred)
	_proba=NN.predict_proba(self._xtest)
	ypredict=toNum(_proba,probability)
	#core=getscore(self._ytest, predict)
	cnf_matrix=confusion_matrix(ytest,ypredict)
	t1=sum(cnf_matrix[0,:])#真实做空机会
	t2=sum(cnf_matrix[1,:])
	t3=sum(cnf_matrix[2,:])#真实做多机会
	p1=sum(cnf_matrix[:,0])#判断做空次数
	p2=sum(cnf_matrix[:,1])
	p3=sum(cnf_matrix[:,2])#判断做空次数
	P1T=cnf_matrix[0,0]#判断做空为正确次数
	P1F=cnf_matrix[2,0]#判断做空实际上涨次数
	P3T=cnf_matrix[2,2]#判断做多为正确次数
	P3F=cnf_matrix[0,2]#判断做多实际下跌次数
	if(p1<1):p1=1
	if(t1<1):t1=1
	if(p3<1):p3=1
	if(t3<1):t3=1
	p1tRate=(P1T-P1F)/p1;#判断做空正确概率
	# if p1tRate<0:
		# p1tRate=0;
	p1Hit=P1T/t1;#做空机会命中率
	p3tRate=(P3T-P3F)/p3;#判断做多正确概率
	p3Hit=P3T/t3;#做多机会命中率
	#print('做空正确概率:%.2f %%'%(100*p1tRate))
	#print('做空命中概率:%.2f %%'%(100*p1Hit))
	#print('做多正确概率:%.2f %%'%(100*p3tRate))
	#print('做多命中概率:%.2f %%'%(100*p3Hit))
	score=(0.682*p1tRate+0.318*p1Hit)*0.5+(0.682*p3tRate+0.318*p3Hit)*0.5
	#print('策略评分为:%.2f' % (score*100))
	return score	
	
	
def getname(data):
	data = [ int (data) for data in data]
	ids = list(set(data))
	# for s in ids:
	 # list.append(s)
	return np.array(ids)

def mltest(xtrain,ytrain,xtest,ytest, name):
	# xtrain=arg.xtrain
	# ytrain=arg.ytrain
	# xtest=arg.xtest
	# ytest=arg.ytest
	if name=="GaussianNB":
	 from sklearn.naive_bayes import GaussianNB

	 clf = GaussianNB()
	 clf.fit(xtrain, ytrain)
	 predict=clf.predict(xtest)
	 #confusion_matrix(y_true=ytest, y_pred=predict)
	elif name=="RandomForest":
	 from sklearn.ensemble import RandomForestClassifier
	 clf = RandomForestClassifier(max_depth=2, random_state=0)
	 clf.fit(xtrain, ytrain)
	 predict=clf.predict(xtest)
	 #confusion_matrix(y_true=ytest, y_pred=predict)
	else:
	 from sklearn import svm
	 classifier = svm.SVC(kernel='linear', C=0.01)
	 predict = classifier.fit(xtrain, ytrain).predict(xtest)
	print(getscoreTrade(ytest, predict))#predictprob=predict_proba(xtest);
	 
	 
	 
#策略保存
def storestrategy(summaries,path='NB_rb3.pkl'):#,summariesx,summariesy,scaler
	# 创建HDF5文件
	file = open(path,'wb')
	# 写入
	pkl.dump(summaries, file)
	# pkl.dump(summariesx,file)
	# pkl.dump(summariesy,file)
	# pkl.dump(scaler,file)
	# file.create_dataset('summaries', data =np.array(summaries))
	# file.create_dataset('summariesx', data = np.array(summariesx))
	# file.create_dataset('summariesy',data = np.array(summariesy))
	file.close()
	return


#策略加载
def loadstrategy(path='NB_rb3.pkl'):
	pkl_file = open(path, 'rb')
	NN=pkl.load(pkl_file)
	pkl_file.close()
	return NN


def datatrainTree(inputtrain,outputtrain,inputtest,outputtest):
    from sklearn import tree
    clf =tree.DecisionTreeClassifier()
    clf = clf.fit(inputtrain, outputtrain)
    accuracy=clf.score(inputtest, outputtest)   
    # accuracy = cal_accuracy(predict,testOutput)
    print('percent: {:.4%}'.format(accuracy))
    return clf	

#策略预测	
def NNpredict(data,NN,times):
	pdata=cl.dataCtimepd(data,times)
	input=pdata.ix[:,:-1].values
	output=NN.predictprob(input)
	outreal=pdata.ix[:,-1]
	
	pout=pd.concat([outreal, pd.DataFrame(outreal,columns=['down','stay','up'])])
	pout.ix[:,1:]=output
	#score=getscoreTrade(outreal,output)
	return pout

#用于多分类的Adaboost方法
def datatrainAdaBoost(trainInput,trainOutput,testInput,testOutput):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
	# SAMME.R uses the probability estimates to update the additive model
    bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1)
    bdt_real.fit(trainInput, trainOutput)#训练
    score1=bdt_real.score(testInput, testOutput)
	#SAMME uses the classifications only. 
    bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1.5, algorithm="SAMME")
    bdt_discrete.fit(trainInput, trainOutput)
    score2=bdt_discrete.score(testInput, testOutput)
    # predict= bdt.predict(testInput)
    if score1>score2:
      accuracy=score1
      result=bdt_real
    else:
      accuracy=score2
      result=bdt_discrete
    # accuracy = cal_accuracy(predict,testOutput)
    print('percent: {:.4%}'.format(accuracy))
    return result
	
	#用于多分类的GradientBoost方法
def datatrainGradientBoost(trainInput,trainOutput,testInput,testOutput):
    from sklearn.ensemble import GradientBoostingClassifier
	# SAMME.R uses the probability estimates to update the additive model
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(trainInput, trainOutput)
    accuracy=clf.score(testInput, testOutput)   
    # accuracy = cal_accuracy(predict,testOutput)
    print('percent: {:.4%}'.format(accuracy))
    return clf
	
def cal_accuracy(predict_y, test_y):
    '''计算预测的准确性
    input: 
            predict_y(mat):预测的标签
            test_y(mat):测试的标签
    output: accuracy(float):预测的准确性
    '''
    n_samples = np.shape(test_y)[0] # 样本的个数
    correct = 0.0
    for i in range(n_samples):
        
        # 判断每一个样本的预测值与真实值是否一致
        if predict_y[i] == test_y[i]:
            correct += 1
    return correct / n_samples

	
	
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
	# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      # title='Normalized confusion matrix')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()