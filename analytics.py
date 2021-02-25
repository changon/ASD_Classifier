from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve
import numpy as np
import torch

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
	if tau.shape[0] > 1:
		tau=tau.flat[0]
	return tau

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

def getAverage(array):
	return sum(array)/len(array)

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
