from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss

def resampleData(x, y, samplingType="under", samplingMethod="Random"):
	if samplingType=="under" and samplingMethod=="Random":
		x, y=RandomUnderSampler().fit_resample(x,y)
	return x,y

def preprocess(data, outcomes, params):
	#unwrap params
	method=params["method"]
	uni=params["uni"]
	cor=params["cor"]
	prep=params["prep"]
	topN=params["topN"]

	ogCols=data.columns
	
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
    
	cols=""
	if uni:
		selector=SelectKBest(chi2, k=topN)
		selector.fit(data, outcomes)
		cols=selector.get_support(indices=True)
		data=data.iloc[:, cols]
		ogCols=ogCols[cols]
    
        # now scale/standardize etc.
        #standaradize
	if prep=="standardize":
		data=preprocessing.scale(data)
		data=pd.DataFrame(data, columns=ogCols)
		#data=scalar.transform(data)
	elif prep=="scale":
		data=preprocessing.MinMaxScaler().fit_transform(data)
		data=pd.DataFrame(data, columns=ogCols)
    
	print('after preprocess, data dim is: '+str(data.shape))
	return data, cols

