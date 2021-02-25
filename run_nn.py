from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import torch
print(torch.__version__)
import torch.nn.functional as F
from torch import nn, optim

import os, sys, random
from random import randrange

from preprocessData import preprocess, resampleData
from model import classifierNetwork
from analytics import getAverage, getLoss
from writeTree import write_tree

def printMetric(metrs, epoch, rType, decimal_places=3):
	s=rType+" - Epoch: "+ str(epoch)+". "
	for key in metrs:
		avg=getAverage(metrs[key])
		avg=round_tensor(avg, decimal_places)
		s+=key+": "+str(avg)+", "
	s=s[0:len(s)-2]
	print(s)
	return

def round_tensor(t, decimal_places=3):
	return round(t.item(), decimal_places)

def initMetric():
	return {"loss":[], "acc": [], "auc":[], "sens": [], "spec": []}

def getOptimizer(mSetting, params, alpha):
	optimizer=optim.SGD(params, lr=alpha)
	if mSetting=="Adam":
		optimizer=optim.Adam(params, lr=alpha)
	return optimizer

def train(xtrain, ytrain, xtest, ytest, modelSetting, nEpochs):
	trainMetrics=initMetric()
	testMetrics=initMetric()

	model=classifierNetwork(modelSetting)

	criterion=nn.BCELoss()
	optimizer=getOptimizer(modelSetting["optimizer"], list(model.parameters()), modelSetting["lr"])
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	X_train=xtrain.to(device)
	y_train=ytrain.to(device)
	X_test=xtest.to(device)
	y_test=ytest.to(device)
	
	model=model.to(device)
	criterion=criterion.to(device)

	finalTau="NONE SET"
	tlosses=[]
	for epoch in range(nEpochs):
		optimizer.zero_grad()
		trainMetrics, train_loss, tau = getLoss(X_train, y_train, model, criterion, trainMetrics)
		#trainStr="Train"
		#printMetric(trainMetrics, epoch, trainStr)
		tlosses.append(trainMetrics)
	
		train_loss.backward()
		optimizer.step()

		if epoch==nEpochs-1:
			finalTau=tau

		#testMetrics, test_loss, roc_data = getLoss(X_test, y_test, model, criterion, testMetrics)
		#printMetric(testMetrics, epoch, "Test")
	return model, criterion, device, finalTau, tlosses

def test(xtest, ytest, model, criter, device, tau, typeStr):
	X_test=xtest.to(device)
	y_test=ytest.to(device)
	testMetrics=initMetric()
	print(tau)
	testMetrics, loss, tau2=getLoss(X_test, y_test, model, criter, testMetrics, tau)
	printMetric(testMetrics, "", typeStr)
	return testMetrics, loss

def printMetaInfo(model, tau, nrounds, tlosses, lr, samplingType, useSubset, optString):
	print("NN Results, trained via "+str(nrounds)+" epochs.")
	trainStr="\tTrain"
	for i in range(len(tlosses)):
		trainMetrics=tlosses[i]
		printMetric(trainMetrics, i+1, trainStr)
	print("\tNEpochs: "+str(nrounds))
	print("\tLearning Rate: "+str(lr))
	print("\tTau: "+str(tau))
	print("\tTraining on Subset: "+str(useSubset))
	print("\tOptimizer: "+str(optString))
	model.printMetaInfo()

def runInstance(xtrain, ytrain, xtest, ytest, modelSetting, nEpochs):
	print("Validating model setting i")
	print("\tTraining")
	model, criterion, device, tau, tlosses=train(xtrain, ytrain, xtest, ytest, modelSetting, nEpochs)

	print("\tTesting")
	validationStr="\tValidation test"
	testMetrics,loss=test(xtest, ytest, model, criterion, device, tau, validationStr)

	print("")
	return model, testMetrics, criterion, device, tau, tlosses

######### NOW actually do stuff
#preprocess data
seedVal=533
random.seed(seedVal)

ncols=int(sys.argv[1])
nRounds=int(sys.argv[2])
ntree=int(sys.argv[3])#100
testP=float(sys.argv[4])#.2
adjustT=bool(sys.argv[5])
lr=float(sys.argv[6])
useSubset=bool(sys.argv[7])
samplingType=str(sys.argv[8])

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

#set up preprocess params
topNs=[10,100,500,1000]
prep="standardize"
cor=False
uni=useSubset
method="none"

params={}
params["prep"]=prep
params["cor"]=cor
params["uni"]=uni
params["method"]=method
params["topN"]=topNs[2]
params["nEpochs"]= nRounds#sloppy to add it here,..

# before transforming, write out decision tree
write_tree(trainData, trainOutcomes, params)

#scale data, index if uni is true
trainData, mask=preprocess(trainData, trainOutcomes, params)
if uni:
	testData=testData.iloc[:, mask]
	validationData=validationData.iloc[:, mask]
testData, mask=preprocess(testData, testOutcomes, params)
validationData, mask=preprocess(validationData, validationOutcomes, params)

#resample train
trainData, trainOutcomes=resampleData(trainData, trainOutcomes)

printIt=False
np.set_printoptions(precision=10)

def transformData(x, y):
	x=x.values.astype(np.float32)
	x=torch.from_numpy(x).float()
	y=y.values.astype(np.float32)
	y=torch.squeeze(torch.from_numpy(y).float())
	return x,y

# now prepare in tensors
trainData, trainOutcomes=transformData(trainData, trainOutcomes)
validationData, validationOutcomes=transformData(validationData, validationOutcomes)
testData, testOutcomes=transformData(testData, testOutcomes)

print(trainData.size(), trainOutcomes.size())
print(validationData.size(), validationOutcomes.size())
print(testData.size(), testOutcomes.size())

print(valInds)

modelSetting={} 
modelSetting["outDim"]=1
modelSetting["nhidden"]=1
modelSetting["lr"]=lr
modelSetting["optimizer"]="Adam"
modelSetting["inputDim"]=trainData.shape[1]
modelSetting["hiddenDims"]=[]
h1=int((modelSetting["inputDim"]+1)/2)
modelSetting["hiddenDims"]=[h1]

import math
hiddens=[h1, 10, 500, int(modelSetting["inputDim"]/2), int(math.sqrt(modelSetting["inputDim"]))]
valMetric="auc"

def performValidation(hiddens, xtrain, ytrain, xtest, ytest, modelSetting, nEpochs, valMetric):
	bestModel=""
	bestResult={}
	bestResult[valMetric]=[-1]
	bestTau=-1
	fDevice=""
	crit=""
	for nhidden in hiddens:
		modelSetting["hiddenDims"]=[nhidden]
		model, testResults, criterion, device, tau, tlosses=runInstance(xtrain, ytrain, xtest, ytest, modelSetting, nEpochs)
		print(tau)
		if getAverage(testResults[valMetric]) > getAverage(bestResult[valMetric]):
			fDevice=device
			bestResult=testResults
			bestModel=model
			bestTau=tau
			crit=criterion
	return model, crit, fDevice, bestTau, tlosses

print("Begin Validation\n")
model, criter, device, tau, tlosses=performValidation(hiddens, trainData, trainOutcomes, validationData, validationOutcomes, modelSetting, nRounds, valMetric)
print("Finished Validation\n")
print("Now testing\n")
printMetaInfo(model, tau, nRounds, tlosses, lr, samplingType, useSubset, modelSetting["optimizer"])
testMetrics, loss=test(testData, testOutcomes, model, criter, device, tau, "Final Test")
