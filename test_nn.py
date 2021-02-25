from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
print(torch.__version__)
import torch.nn.functional as F
from torch import nn, optim

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import random
import os
import sys

from random import randrange
seedVal=533
random.seed(seedVal)

ncols=int(sys.argv[1])
nRounds=int(sys.argv[2])
ntree=int(sys.argv[3])#100
testP=float(sys.argv[4])#.2
adjustT=bool(sys.argv[5])
lr=float(sys.argv[6])

trainData=pd.read_csv("../microarrayASD.csv")
trainOutcomes=pd.read_csv("../microarrayASDLabels.csv")
trainOutcomes[trainOutcomes=="TD"]="0"
trainOutcomes[trainOutcomes=="proband"]="1"

print('train data')
print(trainData.shape)
print(trainOutcomes.shape)

testData=pd.read_csv("../microarrayASD_Test.csv")
testOutcomes=pd.read_csv("../microarrayASDLabels_Test.csv")
testOutcomes[testOutcomes=="TD"]="0"
testOutcomes[testOutcomes=="proband"]="1"

# split into separate validaiton set here
nValidationSize=35#50
nrow=testOutcomes.shape[0]
#valInds=random.sample(range(0, nrow), nValidationSize)
valInds=[48, 21, 44, 23, 16, 55, 26, 42, 3, 31, 22, 18, 11, 24, 8, 38, 7, 54, 1, 19, 35, 64, 13, 39, 32, 9, 62, 60, 6, 0, 50, 49, 15, 59, 27]
validationData=testData.iloc[valInds, :]
validationOutcomes=testOutcomes.iloc[valInds]

print('val data')
print(validationData.shape)
print(validationOutcomes.shape)

print('test data before and after')
print(testData.shape)
print(testOutcomes.shape)

testData=testData.drop(index=testData.index[valInds])
testOutcomes=testOutcomes.drop(index=testOutcomes.index[valInds])

print(testData.shape)
print(testOutcomes.shape)

def preprocess(data, outcomes, params):
	#unwrap params
	method=params["method"]
	uni=params["uni"]
	cor=params["cor"]
	prep=params["prep"]
	topN=params["topN"]

	#undergo transforms,...
	if method=="var":
		p=.1
		sel = VarianceThreshold(threshold=(p * (1 - p)))
		data=sel.fit_transform(data)
	
	data=pd.DataFrame(data)
	
	# next phase of stat cleaning
	thres=.75
	if cor:
		cor_matrix = data.corr().abs()
		upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
		to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > thres)]
		data=data.drop(data.columns[to_drop], axis=1)
	
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2
	cols=""
	if uni:
		selector=SelectKBest(chi2, k=topN)
		selector.fit(data, outcomes)
		cols=selector.get_support(indices=True)
		data=data.iloc[:, cols]
	
	# now scale/standardize etc.
	#standaradize
	if prep=="standardize":
		data=preprocessing.scale(data)
		#data=scalar.transform(data)
	elif prep=="scale":
		data=preprocessing.MinMaxScaler().fit_transform(data)
	
	print('after preprocess, data dim is: '+str(data.shape))
	return data, cols
	
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss

# test on validation via multi rounds
def runValidation(nRounds, ogxtrain, ogytrain, xtest, ytest, params):
	testP=.03
	metrics={"auc": [], "acc": [] , "spec": [], "sens":[]}

	# setup preprocess params and take top n
	ogxtrain, mask=preprocess(ogxtrain, ogytrain, params)
	xtest=xtest.iloc[:, mask]

	#testData, m=preprocess(testData, testOutcomes)
	for i in range(nRounds):
		print("EVALUATING ROUND "+str(i))
		xtrain, ytrain = RandomUnderSampler().fit_resample(ogxtrain, ogytrain) 

		# build clf
		model=RandomForestClassifier(n_estimators=ntree)
		model.fit(xtrain, ytrain)
	
		ypred=model.predict(xtest)
		yprobs=model.predict_proba(xtest)
		acc=accuracy_score(ytest, ypred) 
	
		auc=roc_auc_score(ytest, yprobs[:, 1])
	
		cm=confusion_matrix(ytest, ypred, labels=["0","1"])
		tcm=cm	
		sens = cm[1,1] /(cm[1,1]+cm[1,0])#TP/(TP+FN)
		spec = cm[0,0] /(cm[0,0]+cm[0,1]) #TN/(TN+FP)
	
		metrics["acc"].append(acc)
		metrics["auc"].append(auc)
		metrics["sens"].append(sens)
		metrics["spec"].append(spec)
	return metrics, mask

def printRes(vec):
	mu=np.mean(vec)
	std=np.std(vec)
	print(str(mu)+' ± ' + str(std))

def printAllRes(n, metrics):
	print("nvar: "+str(n))
	print("Accuracy")
	printRes(metrics["acc"])
	print("AUC")
	printRes(metrics["auc"])
	print("Sens")
	printRes(metrics["sens"])
	print("Spec")
	printRes(metrics["spec"])

def stitchvalues(cwd, p, nevals, outfile):
	filepath=cwd+"/"+outfile+".parmams.p:"+str(p)+".nevals:"+str(nevals)+".csv"
	return filepath

def wrapup(imps_df, metr, n, mask, trainData, trainOutcomes):
	print("Final Results")
	printAllRes(n, metr)

	outputFile="importances"+str(marker)
	pwd=os.getcwd()
	outputFilePathImps=stitchvalues(pwd, n, nRounds, outputFile)
	print("\twriting "+outputFilePathImps)
	imps_df.T.to_csv(outputFilePathImps) #, index=False)
	
	subTrainData=trainData.iloc[:,mask]
	clf=DecisionTreeClassifier().fit(subTrainData, trainOutcomes)
	plt.rcParams["figure.figsize"] = (120, 120)
	tree.plot_tree(clf, feature_names=list(subTrainData.columns.values), class_names=["0", "1"], filled=True)
	plt.show()
	dtName="decision_tree"+str(marker)+".png"
	ffname=str(n)+"."+str(marker)+"_"+dtName
	print("\twriting "+ffname)
	plt.savefig(ffname)


def getOptimalThreshold(ys, probs,labels=[0,1]):
	#use youden
	fpr, tpr, thresholds = roc_curve(ys.astype(int), probs, pos_label=1)
	vals=[]
	n=thresholds.shape[0]
	for i in range(n):
		tau=thresholds[i]
		predicted=[labels[1] if p >= tau else labels[0] for p in probs]
		cm=confusion_matrix(ys, predicted, labels=labels)
		sens = cm[1,1] /(cm[1,1]+cm[1,0])#TP/(TP+FN)
		spec = cm[0,0] /(cm[0,0]+cm[0,1]) #TN/(TN+FP)
		vals.append(sens+spec)
	bestIndex=np.where(vals==max(vals))
	tau=thresholds[bestIndex]
	return tau,
	

def evalVars(xtrain, ytrain, xval, yval, xtest, ytest, params, varsToKeep, adjustThreshold=False):
	xtrain=xtrain[varsToKeep]
	xtest=xtest[varsToKeep]
	print("data dims")
	print(xtrain.shape)
	print(xtest.shape)

	xtrain2, ytrain2 = RandomUnderSampler().fit_resample(xtrain, ytrain) 
	
	model=RandomForestClassifier(n_estimators=ntree) #, random_state =seedVal)
	model.fit(xtrain2, ytrain2)
	
	ypred=model.predict(xtest)
	yprobs=model.predict_proba(xtest)
	acc=accuracy_score(ytest, ypred)
	auc=roc_auc_score(ytest, yprobs[:, 1])

	if adjustThreshold:
		optimalThres=getOptimalThreshold(ytest, yprobs[:,1])
	
	cm=confusion_matrix(ytest, ypred, labels=["0","1"])
	sens = cm[1,1] /(cm[1,1]+cm[1,0])#TP/(TP+FN)
	spec = cm[0,0] /(cm[0,0]+cm[0,1]) #TN/(TN+FP)
	
	print("AUC: "+str(auc))
	print("Acc: "+str(acc))
	print("sens: "+str(sens))
	print("spec: "+str(spec))
	return auc, acc, sens, spec
	
def runNN():
	return

######### NOW actually do stuff
#set up preprocess params
#preprocess params
topNs=[10,100,500,1000]
prep="standardize"
#"scale"
#"None"
cor=False
uni=False
method="none"

params={}
params["prep"]=prep
params["cor"]=cor
params["uni"]=uni
params["method"]=method
params["topN"]=500

#scale data
trainData, mask=preprocess(trainData, trainOutcomes, params)
uni=False
params["uni"]=uni
if uni:
	testData=testData.iloc[:, mask]
testData, mask=preprocess(testData, testOutcomes, params)

topN=10
marker=3

printIt=False
np.set_printoptions(precision=10)
tcm="" # for cm

# now test
# setup preprocess params and take top n
varsToKeep=["163732", "4306", "6256", "401470", "3985", "5816", "6037", "1523", "649801", "6228", "222166", "647436", "3838", "57535", "728715", "10635", "7033"]

print(valInds)

#trainData=trainData.values.astype(np.float32)
xtrain=torch.from_numpy(trainData).float()
trainOutcomes=trainOutcomes.values.astype(np.float32)
ytrain=torch.squeeze(torch.from_numpy(trainOutcomes).float())

#testData=testData.values.astype(np.float32)
xtest=torch.from_numpy(testData).float()
testOutcomes=testOutcomes.values.astype(np.float32)
ytest=torch.squeeze(torch.from_numpy(testOutcomes).float())

print(xtrain.size(), ytest.size())
print(xtest.size(), ytest.size())

#https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/

class classifierNetwork(nn.Module):
	def __init__(self, n_features, nhidden, nhidden2):
		super(classifierNetwork, self).__init__()
		self.fc1=nn.Linear(n_features, nhidden)
		torch.nn.init.xavier_uniform(self.fc1.weight)

		self.fc2=nn.Linear(nhidden,nhidden2)
		torch.nn.init.xavier_uniform(self.fc2.weight)

		self.fc3=nn.Linear(nhidden2,1)
		torch.nn.init.xavier_uniform(self.fc3.weight)

		self.l1=n_features
		self.l2=nhidden
		self.l3=nhidden2

	def forward(self, x):
		x=self.fc1(x)
		x=torch.sigmoid(x)
		x=self.fc2(x)
		x=torch.sigmoid(x)
		x=self.fc3(x)
		x=torch.sigmoid(x)
#extra isgmoid?
		return torch.sigmoid(x)

class classifierNetworkSingle(nn.Module):
	def __init__(self, n_features, nhidden):
		super(classifierNetworkSingle, self).__init__()
		self.fc1=nn.Linear(n_features, nhidden)
		torch.nn.init.xavier_uniform(self.fc1.weight)

		self.fc2=nn.Linear(nhidden,1)
		torch.nn.init.xavier_uniform(self.fc2.weight)

		self.l1=n_features
		self.l2=nhidden

	def forward(self, x):
		x=self.fc1(x)
		x=torch.sigmoid(x)
		x=self.fc2(x)
		return torch.sigmoid(x)
	def printMetaInfo(self):
		print("Network info: 1 hidden layer. Hidden layer size: "+str(self.l2))


def calculate_basics(y_true, y_pred, threshold=.5, labels=[0,1]):
	probs=y_pred.detach().numpy()
	predicted=[labels[1] if p >= threshold else labels[0] for p in probs]

	metric={}
	y_true=y_true.detach().numpy().astype(int)
	cm=confusion_matrix(y_true, predicted, labels=labels)
	metric["sens"] = cm[1,1] /(cm[1,1]+cm[1,0])#TP/(TP+FN)
	metric["spec"] = cm[0,0] /(cm[0,0]+cm[0,1]) #TN/(TN+FP)
	corrects=(cm[1,1]+cm[0,0])
	metric["acc"] = corrects/float(cm[0,1]+cm[1,0]+corrects)
	return metric

def getLoss(x, y, mod, crit, metrs, tau=-1):
	y_pred=mod(x)
	y_pred=torch.squeeze(y_pred)
	loss=crit(y_pred, y)
	
	basics=calculate_basics(y, y_pred)
	auc=roc_auc_score(y.detach().numpy(), y_pred.detach().numpy())

	if tau==-1:
		tau=getOptimalThreshold(y.detach().numpy(), y_pred.detach().numpy())
	else:
		basics=calculate_basics(y, y_pred, tau)

	metrs["acc"].append(basics["acc"])
	metrs["sens"].append(basics["sens"])
	metrs["spec"].append(basics["spec"])
	metrs["auc"].append(auc)
	metrs["loss"].append(loss)
	return metrs, loss, tau

def getAverage(array):
	return sum(array)/len(array)

def printMetric(metrs, epoch, rType):
	s=rType+" - Epoch: "+ str(epoch)+". "
	for key in metrs:
		avg=getAverage(metrs[key])
		s+=key+": "+str(avg)+", "
	s=s[0:len(s)-2]
	print(s)
	return

def round_tensor(t, decimal_places=3):
	return round(t.item(), decimal_places)

def initMetric():
	return {"loss":[], "acc": [], "auc":[], "sens": [], "spec": []}

def train(xtrain, ytrain, xtest, ytest, alpha=.01):
	trainMetrics=initMetric()
	testMetrics=initMetric()
	inpDim=xtrain.shape[1]

	#h1=int(inpDim/2)
	#h2=int(h1/2)
	#model=classifierNetwork(inpDim, h1, h2)#old results on 500 subset had 500, 250, 100 (vs. 500 , 250, 125)

	h1=int((inpDim+1)/2)
	model=classifierNetworkSingle(inpDim, h1)
	criterion=nn.BCELoss()
	params=list(model.parameters())
	#optimizer=optim.Adam(params, lr=alpha)
	optimizer=optim.SGD(params, lr=alpha)
	device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	X_train=xtrain.to(device)
	y_train=ytrain.to(device)
	X_test=xtest.to(device)
	y_test=ytest.to(device)
	
	model=model.to(device)
	criterion=criterion.to(device)
	nEpochs=nRounds
	finalTau="NONE SET"
	for epoch in range(nEpochs):
		optimizer.zero_grad()
		trainMetrics, train_loss, tau = getLoss(X_train, y_train, model, criterion, trainMetrics)
		trainStr="Train"
		printMetric(trainMetrics, epoch, trainStr)
	
		train_loss.backward()
		optimizer.step()

		if epoch==nEpochs-1:
			finalTau=tau

		#testMetrics, test_loss, roc_data = getLoss(X_test, y_test, model, criterion, testMetrics)
		#printMetric(testMetrics, epoch, "Test")
	return model, criterion, device, finalTau

def test(xtest, ytest, model, criter, device, tau):
	X_test=xtest.to(device)
	y_test=ytest.to(device)
	testMetrics=initMetric()
	testMetrics, loss, t=getLoss(X_test, y_test, model, criter, testMetrics, tau)
	printMetric(testMetrics, "", "Final Test")

def printMetaInfo(model, tau, nrounds):
	print("NN Results, trained via "+str(nrounds)+" epochs.")
	print("Tau: "+str(tau))
	model.printMetaInfo()

print("Training")
model, criterion, device, tau=train(xtrain, ytrain, xtest, ytest, lr)
print("Testing")
test(xtest, ytest, model, criterion, device, tau)

printMetaInfo(model, tau, nRounds)
